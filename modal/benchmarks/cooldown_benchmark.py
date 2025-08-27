def benchmark_cooldown_impact():
    """Benchmark the impact of the cooldown period on reusing containers."""
    results = BenchmarkResults("Cooldown Impact Performance")
    workspace_id = "benchmark-workspace-cooldown"
    
    # Create executor
    executor = ModalExecutor(workspace_id=workspace_id)
    python_code = get_test_python_code("simple")
    
    print("\nTesting cooldown impact (30s window)...")
    print("First request will trigger a cold start...")
    
    # First run (cold start)
    start_time = time.time()
    try:
        result = executor.execute_remote(
            block_type_name="cooldown_test",
            python_code=python_code,
            inputs={"x": 5, "y": 10},
            workspace_id=workspace_id
        )
        duration = time.time() - start_time
        results.add_result(duration, is_cold_start=True, result=result)
        print(f"Cold start: {duration:.4f} seconds")
    except Exception as e:
        duration = time.time() - start_time
        results.add_result(duration, is_cold_start=True, error=e)
        print(f"Error during cold start: {e}")
    
    # Run a series of requests with different delays
    delays = [1, 5, 10, 15, 20, 25, 29, 31, 35]
    
    for delay in delays:
        print(f"\nWaiting {delay} seconds before next request...")
        time.sleep(delay)
        
        start_time = time.time()
        try:
            result = executor.execute_remote(
                block_type_name="cooldown_test",
                python_code=python_code,
                inputs={"x": 5, "y": 10},
                workspace_id=workspace_id
            )
            duration = time.time() - start_time
            # If delay > 30s (our cooldown window), this is likely a cold start
            is_cold = delay > 30
            results.add_result(duration, is_cold_start=is_cold, result=result)
            print(f"Request after {delay}s delay: {duration:.4f} seconds")
            print(f"Container type: {'Cold start' if is_cold else 'Warm container'}")
        except Exception as e:
            duration = time.time() - start_time
            results.add_result(duration, is_cold_start=delay > 30, error=e)
            print(f"Error during request: {e}")
    
    results.print_summary()
    
    # Create a specific plot for cooldown impact
    delays_under_cooldown = [d for d in delays if d <= 30]
    delays_over_cooldown = [d for d in delays if d > 30]
    
    times_under_cooldown = [r for i, r in enumerate(results.durations[1:]) if delays[i-1] <= 30]
    times_over_cooldown = [r for i, r in enumerate(results.durations[1:]) if delays[i-1] > 30]
    
    plt.figure(figsize=(12, 6))
    plt.axvline(x=30, color='r', linestyle='--', label='Cooldown threshold (30s)')
    
    if delays_under_cooldown and times_under_cooldown:
        plt.scatter(delays_under_cooldown, times_under_cooldown, color='green', label='Within cooldown (warm)')
    
    if delays_over_cooldown and times_over_cooldown:
        plt.scatter(delays_over_cooldown, times_over_cooldown, color='orange', label='Beyond cooldown (cold)')
    
    plt.xlabel('Delay between requests (seconds)')
    plt.ylabel('Execution time (seconds)')
    plt.title('Impact of Cooldown Period on Execution Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig("cooldown_impact.png")
    print("Cooldown impact plot saved as 'cooldown_impact.png'")
    
    return results