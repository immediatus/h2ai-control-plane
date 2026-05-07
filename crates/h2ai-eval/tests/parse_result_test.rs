use h2ai_eval::parse_result;

#[test]
fn pytest_all_passed() {
    let output = "3 passed in 0.12s";
    let (passed_bool, passed, failed, total) = parse_result(output, true);
    assert!(passed_bool);
    assert_eq!(passed, 3);
    assert_eq!(failed, 0);
    assert_eq!(total, 3);
}

#[test]
fn pytest_mixed_pass_fail() {
    let output = "2 passed, 1 failed in 0.45s";
    let (passed_bool, passed, failed, total) = parse_result(output, false);
    assert!(!passed_bool);
    assert_eq!(passed, 2);
    assert_eq!(failed, 1);
    assert_eq!(total, 3);
}

#[test]
fn pytest_all_failed() {
    let output = "3 failed in 0.08s";
    let (passed_bool, passed, failed, total) = parse_result(output, false);
    assert!(!passed_bool);
    assert_eq!(passed, 0);
    assert_eq!(failed, 3);
    assert_eq!(total, 3);
}

#[test]
fn no_counts_falls_back_to_exit_ok_true() {
    // Runner that doesn't emit structured counts but exited 0
    let output = "OK: all checks passed";
    let (passed_bool, passed, failed, total) = parse_result(output, true);
    assert!(passed_bool, "no counts + exit_ok=true → passed");
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
    assert_eq!(total, 0);
}

#[test]
fn no_counts_falls_back_to_exit_ok_false() {
    let output = "ERROR: something went wrong";
    let (passed_bool, ..) = parse_result(output, false);
    assert!(!passed_bool, "no counts + exit_ok=false → failed");
}

#[test]
fn empty_output_exit_ok_false() {
    let (passed_bool, ..) = parse_result("", false);
    assert!(!passed_bool);
}
