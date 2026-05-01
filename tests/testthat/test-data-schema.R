test_that("events schema validation works", {
  ok <- data.frame(patient_id = 1, event = "A", age = 0.5)
  expect_invisible(survivehr_validate_events(ok))

  bad <- data.frame(patient_id = 1, age = 0.5)
  expect_error(survivehr_validate_events(bad), regexp = "event")
})

test_that("targets schema validation works", {
  ok <- data.frame(patient_id = 1, target_event = "A", target_age = 0.8)
  expect_invisible(survivehr_validate_targets(ok))

  bad <- data.frame(patient_id = 1, target_age = 0.8)
  expect_error(survivehr_validate_targets(bad), regexp = "target")
})
