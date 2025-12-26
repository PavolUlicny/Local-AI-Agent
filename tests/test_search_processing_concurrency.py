"""Tests for ThreadSafeState race condition fixes and concurrent operations."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from src.search_context import SearchState
from src.search_processing_async import ThreadSafeState


class TestThreadSafeStateRaceConditions:
    """Test that ThreadSafeState prevents race conditions in concurrent scenarios."""

    def test_concurrent_url_additions_no_duplicates(self) -> None:
        """Test that concurrent URL additions don't create duplicates."""
        state = SearchState()
        safe_state = ThreadSafeState(state)

        # Same URL that will be added concurrently
        test_url = "http://example.com/test"
        num_threads = 10
        results = []

        def add_url():
            # Each thread tries to add the same URL
            was_added = safe_state.check_and_add_url(test_url)
            results.append(was_added)

        # Run threads concurrently
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(add_url) for _ in range(num_threads)]
            for future in futures:
                future.result()

        # Only ONE thread should have successfully added the URL
        assert results.count(True) == 1, "Exactly one thread should add the URL"
        assert results.count(False) == num_threads - 1, "Other threads should see it as duplicate"

        # State should contain exactly one URL
        assert len(state.seen_urls) == 1
        assert test_url in state.seen_urls

    def test_concurrent_hash_additions_no_duplicates(self) -> None:
        """Test that concurrent hash additions don't create duplicates."""
        state = SearchState()
        safe_state = ThreadSafeState(state)

        # Same hash that will be added concurrently
        test_hash = "abc123def456"
        num_threads = 10
        results = []

        def add_hash():
            # Each thread tries to add the same hash
            was_added = safe_state.check_and_add_result_hash(test_hash)
            results.append(was_added)

        # Run threads concurrently
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(add_hash) for _ in range(num_threads)]
            for future in futures:
                future.result()

        # Only ONE thread should have successfully added the hash
        assert results.count(True) == 1, "Exactly one thread should add the hash"
        assert results.count(False) == num_threads - 1, "Other threads should see it as duplicate"

        # State should contain exactly one hash
        assert len(state.seen_result_hashes) == 1
        assert test_hash in state.seen_result_hashes

    def test_mixed_operations_are_thread_safe(self) -> None:
        """Test that mixed read/write operations are thread-safe."""
        state = SearchState()
        safe_state = ThreadSafeState(state)

        urls_to_add = [f"http://example.com/{i}" for i in range(50)]
        num_threads_per_url = 5
        total_additions = 0
        lock = threading.Lock()

        def add_multiple_urls():
            nonlocal total_additions
            for url in urls_to_add:
                was_added = safe_state.check_and_add_url(url)
                if was_added:
                    with lock:
                        total_additions += 1

        # Run threads concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(add_multiple_urls) for _ in range(num_threads_per_url)]
            for future in futures:
                future.result()

        # Each URL should have been added exactly once
        assert total_additions == len(urls_to_add), "Each URL should be added exactly once"
        assert len(state.seen_urls) == len(urls_to_add)
        for url in urls_to_add:
            assert url in state.seen_urls

    def test_concurrent_keyword_updates(self) -> None:
        """Test that concurrent keyword updates don't lose data."""
        state = SearchState()
        safe_state = ThreadSafeState(state)

        # Each thread will add different keywords
        num_threads = 10
        keywords_per_thread = 5

        def add_keywords(thread_id: int):
            keywords = {f"keyword_{thread_id}_{i}" for i in range(keywords_per_thread)}
            safe_state.update_keywords(keywords)

        # Run threads concurrently
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(add_keywords, i) for i in range(num_threads)]
            for future in futures:
                future.result()

        # All keywords should be present
        expected_keywords = {f"keyword_{t}_{k}" for t in range(num_threads) for k in range(keywords_per_thread)}
        assert state.topic_keywords == expected_keywords

    def test_get_topic_keywords_returns_copy(self) -> None:
        """Test that get_topic_keywords returns a copy (not shared reference)."""
        state = SearchState()
        safe_state = ThreadSafeState(state)

        # Add initial keywords
        initial_keywords = {"python", "programming", "code"}
        safe_state.update_keywords(initial_keywords)

        # Get keywords in thread 1
        keywords1 = safe_state.get_topic_keywords()

        # Modify the returned set
        keywords1.add("modified")

        # Get keywords in thread 2
        keywords2 = safe_state.get_topic_keywords()

        # Original state should not be affected by modification
        assert "modified" not in keywords2
        assert "modified" not in state.topic_keywords
        assert keywords2 == initial_keywords

    def test_race_condition_prevented(self) -> None:
        """Test the classic race condition scenario is prevented.

        This simulates the exact race condition that was possible before:
        Thread A: has_url() -> False
        Thread B: has_url() -> False (before A adds)
        Thread A: add_url()
        Thread B: add_url() (duplicate!)

        With atomic operations, this can't happen.
        """
        state = SearchState()
        safe_state = ThreadSafeState(state)

        test_url = "http://example.com/race"
        barrier = threading.Barrier(2)  # Synchronize 2 threads
        results = []

        def racy_operation():
            # Synchronize at the barrier to maximize chance of race
            barrier.wait()

            # This atomic operation prevents the race
            was_added = safe_state.check_and_add_url(test_url)
            results.append(was_added)

        # Run exactly 2 threads at the same time
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(racy_operation) for _ in range(2)]
            for future in futures:
                future.result()

        # Verify exactly one succeeded
        assert results.count(True) == 1
        assert results.count(False) == 1
        assert len(state.seen_urls) == 1


class TestDirectStateAccessorAtomicOperations:
    """Test that DirectStateAccessor also implements atomic operations correctly."""

    def test_check_and_add_url_returns_correctly(self) -> None:
        """Test that check_and_add_url returns correct values."""
        from src.search_processing import DirectStateAccessor

        state = SearchState()
        accessor = DirectStateAccessor(state)

        # First add should return True (new)
        assert accessor.check_and_add_url("http://example.com") is True

        # Second add should return False (duplicate)
        assert accessor.check_and_add_url("http://example.com") is False

        # Different URL should return True (new)
        assert accessor.check_and_add_url("http://other.com") is True

    def test_check_and_add_hash_returns_correctly(self) -> None:
        """Test that check_and_add_result_hash returns correct values."""
        from src.search_processing import DirectStateAccessor

        state = SearchState()
        accessor = DirectStateAccessor(state)

        # First add should return True (new)
        assert accessor.check_and_add_result_hash("hash1") is True

        # Second add should return False (duplicate)
        assert accessor.check_and_add_result_hash("hash1") is False

        # Different hash should return True (new)
        assert accessor.check_and_add_result_hash("hash2") is True


class TestThreadSafeStateBackwardCompatibility:
    """Test that old individual operations still work for backward compatibility."""

    def test_individual_has_and_add_operations(self) -> None:
        """Test that individual has/add operations still work."""
        state = SearchState()
        safe_state = ThreadSafeState(state)

        # Test has_url and add_url
        assert safe_state.has_url("http://example.com") is False
        safe_state.add_url("http://example.com")
        assert safe_state.has_url("http://example.com") is True

        # Test has_result_hash and add_result_hash
        assert safe_state.has_result_hash("hash123") is False
        safe_state.add_result_hash("hash123")
        assert safe_state.has_result_hash("hash123") is True

    def test_update_keywords_still_works(self) -> None:
        """Test that update_keywords operation still works."""
        state = SearchState()
        safe_state = ThreadSafeState(state)

        keywords1 = {"python", "code"}
        safe_state.update_keywords(keywords1)
        assert state.topic_keywords == keywords1

        keywords2 = {"java", "script"}
        safe_state.update_keywords(keywords2)
        assert state.topic_keywords == keywords1 | keywords2


class TestProtocolCompliance:
    """Test that both accessors properly implement the StateAccessor protocol."""

    def test_thread_safe_state_implements_protocol(self) -> None:
        """Test that ThreadSafeState implements all required protocol methods."""

        state = SearchState()
        safe_state = ThreadSafeState(state)

        # Check all protocol methods exist and are callable
        assert callable(safe_state.check_and_add_url)
        assert callable(safe_state.check_and_add_result_hash)
        assert callable(safe_state.has_url)
        assert callable(safe_state.has_result_hash)
        assert callable(safe_state.add_url)
        assert callable(safe_state.add_result_hash)
        assert callable(safe_state.update_keywords)
        assert callable(safe_state.get_topic_keywords)

    def test_direct_state_accessor_implements_protocol(self) -> None:
        """Test that DirectStateAccessor implements all required protocol methods."""
        from src.search_processing import DirectStateAccessor

        state = SearchState()
        accessor = DirectStateAccessor(state)

        # Check all protocol methods exist and are callable
        assert callable(accessor.check_and_add_url)
        assert callable(accessor.check_and_add_result_hash)
        assert callable(accessor.has_url)
        assert callable(accessor.has_result_hash)
        assert callable(accessor.add_url)
        assert callable(accessor.add_result_hash)
        assert callable(accessor.update_keywords)
        assert callable(accessor.get_topic_keywords)
