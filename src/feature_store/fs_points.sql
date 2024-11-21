-- CTE to calculate weekly points balance and cumulative points for the last 3 weeks
WITH tb_points_d AS (
    SELECT
        idCustomer,
        -- Total points balance for the last 3 weeks
        SUM(pointsTransaction) AS balancePontsW3,

        -- Points balance for the last 2 weeks
        SUM(CASE WHEN dtTransaction >= DATE('{date}', '-14 day') THEN pointsTransaction ELSE 0 END) AS balancePointsW2,

        -- Points balance for the last week
        SUM(CASE WHEN dtTransaction >= DATE('{date}', '-7 day') THEN pointsTransaction ELSE 0 END) AS balancePointsW1,

        -- Cumulative points earned in the last 3 weeks
        SUM(CASE WHEN pointsTransaction > 0 THEN pointsTransaction ELSE 0 END) AS pointsCumulativeW3,

        -- Cumulative points earned in the last 2 weeks
        SUM(CASE WHEN pointsTransaction > 0 AND dtTransaction >= DATE('{date}', '-14 day') THEN pointsTransaction ELSE 0 END) AS pointsCumulativeW2,

        -- Cumulative points earned in the last week
        SUM(CASE WHEN pointsTransaction > 0 AND dtTransaction >= DATE('{date}', '-7 day') THEN pointsTransaction ELSE 0 END) AS pointsCumulativeW1,

        -- Points redeemed in the last 3 weeks (negative transactions)
        SUM(CASE WHEN pointsTransaction < 0 THEN pointsTransaction ELSE 0 END) AS pointsRedeemW3,

        -- Points redeemed in the last 2 weeks
        SUM(CASE WHEN pointsTransaction < 0 AND dtTransaction >= DATE('{date}', '-14 day') THEN pointsTransaction ELSE 0 END) AS pointsRedeemW2,

        -- Points redeemed in the last week
        SUM(CASE WHEN pointsTransaction < 0 AND dtTransaction >= DATE('{date}', '-7 day') THEN pointsTransaction ELSE 0 END) AS pointsRedeemW1

    FROM transactions
    -- Filter to consider transactions within the last 3 weeks
    WHERE dtTransaction < '{date}' AND dtTransaction >= DATE('{date}', '-21 day')
    GROUP BY idCustomer
),

-- CTE to calculate the customer's lifetime points data and account age in days
tb_life AS (
    SELECT
        t1.idCustomer,
        -- Total lifetime points balance
        SUM(t2.pointsTransaction) AS pointsBalance,

        -- Total lifetime cumulative points (earned)
        SUM(CASE WHEN t2.pointsTransaction > 0 THEN t2.pointsTransaction ELSE 0 END) AS pointsCumulativeLife,

        -- Total lifetime redeemed points (negative transactions)
        SUM(CASE WHEN t2.pointsTransaction < 0 THEN t2.pointsTransaction ELSE 0 END) AS pointsRedeemLife,

        -- Account age in days (from the earliest transaction date to reference date)
        CAST(MAX(julianday('{date}') - julianday(t2.dtTransaction)) AS INTEGER) + 1 AS daysLife

    FROM tb_points_d AS t1
    LEFT JOIN transactions AS t2 ON t1.idCustomer = t2.idCustomer
    WHERE t2.dtTransaction < '{date}'
    GROUP BY t1.idCustomer
),

-- Join the weekly points data with lifetime data for each customer
tb_join AS (
    SELECT
        t1.*,
        t2.pointsBalance,
        t2.pointsCumulativeLife,
        t2.pointsRedeemLife,
        -- Average points redeemed per day (lifetime)
        1.0 * t2.pointsRedeemLife / t2.daysLife AS pointsPerDay
    FROM tb_points_d AS t1
    LEFT JOIN tb_life AS t2 ON t1.idCustomer = t2.idCustomer
)

-- Final SELECT to output the combined data with reference date
SELECT
    '{date}' AS dtRef,
    *
FROM tb_join;
