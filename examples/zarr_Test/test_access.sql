#!/usr/bin/env bash

DB="cudaheat"
USER="olddog"
HOST="localhost"
PORT="5433"

RUN_ID="14962e13-1425-43b9-ad1b-333f0bc40e02"

SIZES=(100 500 1000 5000 15000 30000 50000 100000)

echo
echo "========================================"
echo "POSTGRESQL SPATIAL ACCESS BENCHMARK"
echo "========================================"
echo

for N in "${SIZES[@]}"; do

    echo "----------------------------------------"
    echo "N = $N"
    echo "----------------------------------------"

    echo -n "ORDERED: "

    psql \
        -h "$HOST" \
        -p "$PORT" \
        -U "$USER" \
        -d "$DB" \
        -X -q \
        -o /dev/null \
        -c "\timing on" \
        -c "
        SELECT
            e.event_id,
            p.grid_point_index,
            e.start_time,
            e.end_time
        FROM events e
        JOIN points p
            ON p.point_id = e.point_id
        WHERE e.run_id = '$RUN_ID'
        ORDER BY
            p.grid_point_index,
            e.start_time
        LIMIT $N;
        "

    echo -n "RANDOM:  "

    psql \
        -h "$HOST" \
        -p "$PORT" \
        -U "$USER" \
        -d "$DB" \
        -X -q \
        -o /dev/null \
        -c "\timing on" \
        -c "
        SELECT
            e.event_id,
            p.grid_point_index,
            e.start_time,
            e.end_time
        FROM events e
        JOIN points p
            ON p.point_id = e.point_id
        WHERE e.run_id = '$RUN_ID'
        ORDER BY random()
        LIMIT $N;
        "

    echo
done

echo "========================================"
echo "BENCHMARK COMPLETE"
echo "========================================"