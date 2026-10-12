from vllm_tuner.helper.display import DisplayConsole


class TestDisplayConsole:
    def test_display_trial_result(self):
        console = DisplayConsole()
        console.display_trial_result(
            {"Throughput": "10.5", "P95 Latency": "145.0"},
            trial_number=1,
        )

    def test_display_study_summary(self):
        console = DisplayConsole()
        console.display_study_summary(
            {"Trials": "50", "Best Throughput": "12.5"},
            study_name="test-study",
        )

    def test_display_progress_bar(self):
        console = DisplayConsole()
        console.display_progress_bar(25, 50, label="Tuning")

    def test_display_progress_bar_zero(self):
        console = DisplayConsole()
        console.display_progress_bar(0, 0, label="Empty")
