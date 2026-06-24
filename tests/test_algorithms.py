from worker.algorithms import primes, pi_monte_carlo, stress


class TestPrimes:
    def test_is_prime_small_numbers(self):
        assert not primes.is_prime(0)
        assert not primes.is_prime(1)
        assert primes.is_prime(2)
        assert primes.is_prime(3)
        assert not primes.is_prime(4)
        assert primes.is_prime(5)
        assert not primes.is_prime(9)
        assert primes.is_prime(11)

    def test_is_prime_large_prime(self):
        assert primes.is_prime(7919)
        assert primes.is_prime(104729)

    def test_is_prime_large_composite(self):
        assert not primes.is_prime(10000)
        assert not primes.is_prime(7917)

    def test_compute_chunk(self):
        result = primes.compute_chunk(1, 20)
        assert result == [2, 3, 5, 7, 11, 13, 17, 19]

    def test_compute_chunk_empty(self):
        assert primes.compute_chunk(0, 1) == []
        assert primes.compute_chunk(20, 10) == []

    def test_run_rank_0(self):
        result = primes.run({"limit": 100}, 0, 4)
        assert "prime_count" in result
        assert "sample" in result
        assert len(result["sample"]) <= 10
        assert result["prime_count"] >= 0

    def test_run_rank_3_last(self):
        result = primes.run({"limit": 100}, 3, 4)
        assert result["prime_count"] >= 0


class TestPiMonteCarlo:
    def test_run_returns_dict_with_expected_keys(self):
        result = pi_monte_carlo.run({"points": 10000}, 0, 2)
        assert "inside" in result
        assert "total" in result
        assert "pi_approx" in result
        assert result["total"] == 5000

    def test_pi_approx_reasonable(self):
        result = pi_monte_carlo.run({"points": 100000}, 0, 1)
        assert 2.0 < result["pi_approx"] < 4.0

    def test_default_params(self):
        result = pi_monte_carlo.run({}, 0, 1)
        assert result["total"] == 10_000_000  # default points


class TestStress:
    def test_run_returns_dict(self):
        result = stress.run({"iterations": 1000}, 0, 1)
        assert "ops" in result
        assert "elapsed" in result
        assert "ops_per_sec" in result
        assert result["ops"] == 1000

    def test_ops_per_sec_positive(self):
        result = stress.run({"iterations": 1000}, 0, 2)
        assert result["ops_per_sec"] > 0

    def test_default_params(self):
        result = stress.run({}, 0, 1)
        assert result["ops"] == 10_000_000  # default iterations
