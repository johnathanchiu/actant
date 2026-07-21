# Actant development commands

python_version := "3.11"
compose := "docker compose -f docker-compose.yml"

default:
    @just --list

# Sync the local development environment with provider + Temporal extras.
sync:
    uv python install {{python_version}}
    uv sync --python {{python_version}} --extra dev --extra providers

# Manage the packaged local Temporal server (`just server start`).
server command *args="":
    uv run actant server {{command}} {{args}}

# Run Actant tests.
test *args="":
    uv run --extra dev --extra providers python -m pytest {{args}}

# Run lint checks for the repo.
lint:
    uv run --extra dev --extra providers ruff check .

# Type-check the package and tests.
typecheck:
    uv run --extra dev --extra providers pyright actant tests

# Build and validate the distributions without publishing them.
package:
    uv build --clear
    uvx twine check dist/*

# Install demo dependencies (Python server + JS UI).
demo-sync:
    cd examples/demo/server && uv sync
    cd examples/demo/ui && bun install

# Run the demo FastAPI server (deterministic unless ACTANT_PROVIDER is set).
demo-server:
    cd examples/demo/server && ACTANT_PROVIDER="${ACTANT_PROVIDER:-fake}" uv run python -m uvicorn app.main:app --port 8181 --reload

# Run just the demo UI.
demo-ui:
    cd examples/demo/ui && bun run dev

# Start the demo's Postgres in the background.
demo-db-up:
    {{compose}} up -d --remove-orphans demo-postgres

# Stop the demo's Postgres.
demo-db-down:
    {{compose}} stop demo-postgres
    {{compose}} rm -f demo-postgres

# Show what's currently bound on the demo's ports. Helpful for
# diagnosing "why doesn't `just demo` start?".
# Show processes currently listening on demo ports.
demo-status:
    #!/usr/bin/env bash
    set -euo pipefail
    service_name() {
        case "$1" in
            5173) echo "UI (vite)" ;;
            8181) echo "server (uvicorn)" ;;
            27233) echo "Temporal gRPC" ;;
            28233) echo "Temporal UI" ;;
            55435) echo "demo Postgres" ;;
        esac
    }
    any=0
    for port in 5173 8181 27233 28233 55435; do
        line=$(lsof -nP -iTCP:$port -sTCP:LISTEN 2>/dev/null | sed -n '2p' || true)
        if [ -n "$line" ]; then
            cmd=$(echo "$line" | awk '{print $1}')
            pid=$(echo "$line" | awk '{print $2}')
            echo "  :$port ($(service_name "$port")) → $cmd pid=$pid"
            any=1
        fi
    done
    if [ $any -eq 0 ]; then
        echo "  no demo-related ports in use"
    fi

# Preflight before `just demo`. Refuses to start if:
#   - UI/server ports (5173/8181) are taken (uniquely ours).
#   - Docker ports (7233/8233/55432) are taken by something that
#     ISN'T our actant compose stack (e.g. another project's
#     postgres on 55432 — docker can't reuse another container's
#     port mapping).
# If our own actant-* compose containers hold the port, that's fine
# — `docker compose up` will see them and proceed.
# Verify that every port required by the demo is available.
demo-doctor:
    #!/usr/bin/env bash
    set -euo pipefail
    blocking=0
    # Uniquely-ours ports — accept no occupant.
    check_exclusive() {
        local port=$1 service=$2
        if lsof -nP -iTCP:$port -sTCP:LISTEN >/dev/null 2>&1; then
            echo "  ✗ :$port ($service) is in use — blocks the demo"
            blocking=1
        else
            echo "  ✓ :$port ($service) is free"
        fi
    }
    # Docker ports — accept our compose stack, reject anything else.
    check_docker_port() {
        local port=$1 expected_container=$2 service=$3
        if ! lsof -nP -iTCP:$port -sTCP:LISTEN >/dev/null 2>&1; then
            echo "  ✓ :$port ($service) is free"
            return
        fi
        # Find which docker container (if any) owns this published port.
        # ``docker ps`` columns: ID IMAGE CMD CREATED STATUS PORTS NAMES.
        # NAMES is the last column; we scan the whole line for the
        # mapping pattern ":<port>->" and grab the trailing name.
        #
        # We capture docker ps into a variable BEFORE piping to awk so
        # an early-terminating awk doesn't SIGPIPE the docker process —
        # under ``pipefail`` that would exit 141 and abort the whole
        # doctor before later port checks ran.
        local ps_out
        ps_out=$(docker ps 2>/dev/null || true)
        local owner
        owner=$(printf '%s\n' "$ps_out" | awk -v p=":$port->" '$0 ~ p {print $NF; exit}')
        if [ -z "$owner" ]; then
            echo "  ✗ :$port ($service) is in use by a non-docker process"
            blocking=1
        elif [ "$owner" = "$expected_container" ]; then
            echo "  ✓ :$port ($service) held by our compose stack ($owner)"
        else
            echo "  ✗ :$port ($service) is held by $owner — not ours, blocks the demo"
            blocking=1
        fi
    }
    check_exclusive 5173 "UI"
    check_exclusive 8181 "demo server"
    check_docker_port 27233   "actant-temporal-1"          "Temporal gRPC"
    check_docker_port 28233   "actant-temporal-ui-1"       "Temporal UI"
    check_docker_port 55435   "actant-demo-postgres-1"     "demo Postgres"
    if [ $blocking -eq 1 ]; then
        echo
        echo "Fix:"
        echo "  - If a previous \`just demo\` is still running: \`just demo-down\`."
        echo "  - If another project owns the port (e.g. kitsune-world-db on"
        echo "    55432): stop that project's docker container first."
        echo "  - \`just demo-status\` shows what's bound."
        exit 1
    fi
    echo
    echo "Demo ports are clear to start."

# Run the full demo. Refuses to start if any required port is occupied
# (run \`just demo-down\` to clean up a prior instance first, or
# \`just demo-status\` to see what's bound).
#
# Ctrl+C stops the foreground server + UI AND stops the docker
# services we started this run (Temporal, Postgres).
# Run the complete FastAPI, React, Postgres, and Temporal demo.
demo:
    #!/usr/bin/env bash
    set -euo pipefail

    # Preflight: refuse to start if any port is taken. No auto-bumping —
    # that's how the demo accumulated multiple instances and confused
    # tabs landing on stale ports. One demo at a time.
    just demo-doctor

    # Bring up the docker services we own (Temporal + demo-postgres).
    # `rm -f` clears any STOPPED containers from prior failed attempts so
    # they don't hold port reservations.
    {{compose}} rm -f temporal-ui temporal temporal-postgres demo-postgres >/dev/null 2>&1 || true
    {{compose}} up -d --remove-orphans temporal-postgres temporal temporal-ui demo-postgres

    echo "→ Temporal:    localhost:27233 (UI: http://localhost:28233)"
    echo "→ Demo DB:     postgres://actant:actant@localhost:55435/actant_demo"
    echo "→ Demo server: http://localhost:8181"
    echo "→ Demo UI:     http://localhost:5173"

    export ACTANT_TEMPORAL_ADDRESS="localhost:27233"
    export ACTANT_DEMO_DATABASE_URL="postgresql+asyncpg://actant:actant@localhost:55435/actant_demo"
    export VITE_ACTANT_API_BASE="http://localhost:8181"
    export ACTANT_PROVIDER="${ACTANT_PROVIDER:-fake}"

    # ``exec`` inside the subshells means $! is the real uvicorn/vite PID,
    # not the wrapping subshell — kill on cleanup actually reaches them.
    (cd examples/demo/server && exec uv run python -m uvicorn app.main:app --port 8181 --reload) &
    SERVER_PID=$!
    (cd examples/demo/ui && exec bun run dev --port 5173 --strictPort) &
    UI_PID=$!

    cleanup() {
        echo
        echo "→ stopping demo…"
        kill "$SERVER_PID" "$UI_PID" 2>/dev/null || true
        wait "$SERVER_PID" "$UI_PID" 2>/dev/null || true
        {{compose}} stop demo-postgres temporal-ui temporal temporal-postgres >/dev/null 2>&1 || true
    }
    trap cleanup EXIT INT TERM

    wait

# Stop the demo's docker services (used after Ctrl+C if for some reason
# the cleanup trap didn't run, or to reset state).
# Stop and remove the demo's Docker services.
demo-down:
    {{compose}} stop demo-postgres temporal-ui temporal temporal-postgres
    {{compose}} rm -f demo-postgres temporal-ui temporal temporal-postgres
