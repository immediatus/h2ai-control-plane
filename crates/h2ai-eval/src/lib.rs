// Oracle sidecar library — exposes internals for unit testing.
// The binary is in main.rs.

/// Parse pass/fail/total counts from combined test runner output.
///
/// Handles pytest-style summary lines: "3 passed", "2 passed, 1 failed in 0.5s"
/// Falls back to `exit_ok` for runners that don't emit structured counts.
#[must_use]
pub fn parse_result(output: &str, exit_ok: bool) -> (bool, usize, usize, usize) {
    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut found_counts = false;

    for line in output.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        for (i, &part) in parts.iter().enumerate() {
            if i == 0 {
                continue;
            }
            let prev = parts[i - 1];
            match part.trim_end_matches(',') {
                "passed" => {
                    if let Ok(n) = prev.parse::<usize>() {
                        passed = n;
                        found_counts = true;
                    }
                }
                "failed" | "error" | "errors" => {
                    if let Ok(n) = prev.parse::<usize>() {
                        failed += n;
                        found_counts = true;
                    }
                }
                _ => {}
            }
        }
    }

    let total = passed + failed;
    let passed_bool = if found_counts {
        failed == 0 && passed > 0
    } else {
        exit_ok
    };

    (passed_bool, passed, failed, total)
}
