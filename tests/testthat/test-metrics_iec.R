context("Inter-Event Concordance (IEC) Metrics")

# Test survivehr_compute_iec with synthetic data

test_that("survivehr_compute_iec handles basic synthetic data", {
  # Create synthetic risk matrix: 5 transitions, 4 events
  risk_matrix <- matrix(
    c(
      10, 5, 20, 8,    # Transition 1
      15, 25, 10, 12,  # Transition 2
      20, 15, 5, 18,   # Transition 3
      8, 8, 8, 8,      # Transition 4 (identical)
      100, 1, 50, 25   # Transition 5
    ),
    nrow = 5,
    ncol = 4,
    byrow = TRUE
  )

  observed_events <- c(3, 2, 1, 1, 1)

  result <- survivehr_compute_iec(
    risk_scores = risk_matrix,
    observed_events = observed_events
  )

  # Check structure
  expect_is(result, "survivehr_iec")
  expect_is(result, "list")

  # Check required keys
  expect_true("mean_iec" %in% names(result))
  expect_true("n_valid" %in% names(result))
  expect_true("n_total" %in% names(result))
  expect_true("iec_values" %in% names(result))
  expect_true("observed_ranks" %in% names(result))

  # Check values
  expect_equal(result$n_total, 5)
  expect_equal(result$n_valid, 5)
  expect_true(result$mean_iec >= 0 && result$mean_iec <= 1)
  expect_equal(length(result$iec_values), 5)
})


test_that("survivehr_compute_iec validates input dimensions", {
  risk_matrix <- matrix(runif(12), nrow = 3, ncol = 4)
  observed_events <- c(1, 2)  # Wrong length

  expect_error(
    survivehr_compute_iec(risk_matrix, observed_events),
    "Dimension mismatch"
  )
})


test_that("survivehr_compute_iec validates input types", {
  # Non-numeric observed_events
  risk_matrix <- matrix(1:12, nrow = 3, ncol = 4)
  observed_events <- c("a", "b", "c")

  expect_error(
    survivehr_compute_iec(risk_matrix, observed_events),
    "numeric vector"
  )

  # Non-matrix risk_scores
  expect_error(
    survivehr_compute_iec(list(1, 2, 3), c(1, 2, 3)),
    "matrix or data.frame"
  )
})


test_that("survivehr_compute_iec handles data.frame input", {
  risk_df <- data.frame(
    e1 = c(10, 5, 20),
    e2 = c(5, 25, 15),
    e3 = c(20, 10, 5),
    e4 = c(8, 12, 18)
  )
  observed_events <- c(3, 2, 1)

  result <- survivehr_compute_iec(risk_df, observed_events)

  expect_is(result, "survivehr_iec")
  expect_equal(result$n_total, 3)
  expect_equal(result$n_valid, 3)
})


test_that("survivehr_compute_iec handles stratification", {
  risk_matrix <- matrix(
    c(
      10, 5, 20, 8,
      15, 25, 10, 12,
      20, 15, 5, 18
    ),
    nrow = 3,
    ncol = 4,
    byrow = TRUE
  )
  observed_events <- c(1, 2, 3)

  result <- survivehr_compute_iec(
    risk_scores = risk_matrix,
    observed_events = observed_events,
    stratify_by_event = TRUE,
    event_vocabulary = c("HTN", "DM", "MI", "CKD")
  )

  expect_is(result, "survivehr_iec")
  expect_true(result$stratified)
  expect_is(result$by_event, "data.frame")

  # Check stratified columns
  expect_true("event" %in% names(result$by_event))
  expect_true("mean_iec" %in% names(result$by_event))
  expect_true("n_obs" %in% names(result$by_event))
})


test_that("survivehr_compute_iec validates event vocabulary length", {
  risk_matrix <- matrix(runif(12), nrow = 3, ncol = 4)
  observed_events <- c(1, 2, 3)
  event_vocab <- c("A", "B")  # Wrong length

  expect_error(
    survivehr_compute_iec(
      risk_matrix, observed_events,
      stratify_by_event = TRUE,
      event_vocabulary = event_vocab
    ),
    "does not match n_events"
  )
})


test_that("survivehr_compute_iec print method works", {
  risk_matrix <- matrix(runif(12), nrow = 3, ncol = 4)
  observed_events <- c(1, 2, 3)

  result <- survivehr_compute_iec(risk_matrix, observed_events)

  # Print should not error
  expect_silent(print(result))

  # With stratification
  result_strat <- survivehr_compute_iec(
    risk_matrix, observed_events,
    stratify_by_event = TRUE
  )
  expect_silent(print(result_strat))
})


test_that("survivehr_compute_iec IEC values are in [0, 1]", {
  # Generate random risk matrices and check IEC bounds
  for (i in 1:10) {
    risk_matrix <- matrix(runif(50), nrow = 10, ncol = 5)
    observed_events <- sample(1:5, 10, replace = TRUE)

    result <- survivehr_compute_iec(risk_matrix, observed_events)

    expect_true(all(result$iec_values >= 0))
    expect_true(all(result$iec_values <= 1))
    expect_true(result$mean_iec >= 0)
    expect_true(result$mean_iec <= 1)
  }
})


test_that("survivehr_compute_iec perfect ranking gives IEC = 1.0", {
  # Create risk matrix where best-predicted event has highest risk
  risk_matrix <- matrix(
    c(
      10, 5, 30, 8,    # Event 3 has highest risk -> should get IEC = 1.0
      5, 10, 20, 8,    # Event 3 has highest risk -> should get IEC = 1.0
      8, 12, 5, 25     # Event 4 has highest risk -> should get IEC = 1.0
    ),
    nrow = 3,
    ncol = 4,
    byrow = TRUE
  )
  observed_events <- c(3, 3, 4)  # Match the highest-risk events

  result <- survivehr_compute_iec(risk_matrix, observed_events)

  # All IEC values should be 1.0 (perfect prediction)
  expect_true(all(abs(result$iec_values - 1.0) < 1e-6))
  expect_true(abs(result$mean_iec - 1.0) < 1e-6)
})


test_that("survivehr_compute_iec worst ranking gives IEC = 0.0", {
  # Create risk matrix where true event has lowest risk
  risk_matrix <- matrix(
    c(
      30, 25, 10, 20,   # Event 1 observed but has lowest risk -> IEC = 0.0
      25, 30, 20, 10,   # Event 2 observed but has lowest risk -> IEC = 0.0
      20, 25, 30, 10    # Event 4 observed but has lowest risk -> IEC = 0.0
    ),
    nrow = 3,
    ncol = 4,
    byrow = TRUE
  )
  observed_events <- c(1, 2, 4)

  result <- survivehr_compute_iec(risk_matrix, observed_events)

  # All IEC values should be 0.0 (worst prediction)
  expect_true(all(abs(result$iec_values - 0.0) < 1e-6))
  expect_true(abs(result$mean_iec - 0.0) < 1e-6)
})
