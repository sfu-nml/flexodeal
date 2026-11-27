enable_testing()

set(BINARY "${CMAKE_BINARY_DIR}/flexodeal")
set(ROOT   "${CMAKE_SOURCE_DIR}")

# Seconds each smoke test is allowed to run before it is stopped. A test
# passes if the run finishes without errors, or if it is still running
# when this time is up, provided it has started its first time step.
set(SMOKE_RUN_SECONDS 15)

# Find timeout command (gtimeout on macOS, timeout on Linux)
find_program(TIMEOUT_CMD NAMES gtimeout timeout)

if(NOT TIMEOUT_CMD)
  message(WARNING
    "Neither 'timeout' nor 'gtimeout' was found, so the smoke tests are "
    "disabled. On macOS, install them with 'brew install coreutils'.")
  return()
endif()

macro(add_smoke_test NAME COMMAND_STR)
  set(OUT_DIR "${CMAKE_BINARY_DIR}/smoke_out/${NAME}")
  add_test(
    NAME smoke/${NAME}
    COMMAND bash -c "
      mkdir -p ${OUT_DIR}
      ${TIMEOUT_CMD} ${SMOKE_RUN_SECONDS} ${COMMAND_STR} -OUTPUT_DIR=${OUT_DIR} > ${OUT_DIR}/log.txt 2>&1
      CODE=$?
      cat ${OUT_DIR}/log.txt
      if [ $CODE -ne 0 ] && [ $CODE -ne 124 ]; then
        echo \"smoke/${NAME}: run failed with exit code $CODE\"
        exit $CODE
      fi
      if ! grep -q 'Timestep 1 @' ${OUT_DIR}/log.txt; then
        echo \"smoke/${NAME}: no time step started within ${SMOKE_RUN_SECONDS} s\"
        exit 1
      fi
    "
  )
  set_tests_properties(smoke/${NAME} PROPERTIES
    TIMEOUT  30
    LABELS   "smoke"
    WORKING_DIRECTORY  "${ROOT}"
  )
endmacro()

# Default experiment: ./flexodeal with no arguments
add_smoke_test(DEFAULT
  "${BINARY}"
)
