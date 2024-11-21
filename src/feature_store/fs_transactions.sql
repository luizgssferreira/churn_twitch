-- CTE to filter transactions within the last 21 days
WITH cte_transactions AS (
    SELECT *
    FROM transactions
    WHERE dtTransaction < '{date}' 
      AND dtTransaction >= DATE('{date}', '-21 day')
),

-- CTE to calculate frequency of distinct transaction days over the past 21, 14, and 7 days
cte_frequency AS (
    SELECT 
        idCustomer,
        COUNT(DISTINCT DATE(dtTransaction)) AS daysCountD21,
        COUNT(DISTINCT CASE WHEN dtTransaction > DATE('{date}', '-14 day') THEN DATE(dtTransaction) END) AS daysCountD14,
        COUNT(DISTINCT CASE WHEN dtTransaction > DATE('{date}', '-7 day') THEN DATE(dtTransaction) END) AS daysCountD7
    FROM cte_transactions
    GROUP BY idCustomer
),

-- CTE to calculate live session minutes per day, adjusting for Brazil Time (UTC-3)
cte_live_minutes AS (
    SELECT 
        idCustomer,
        DATE(DATETIME(dtTransaction, '-3 hour')) AS sessionDate,
        MIN(DATETIME(dtTransaction, '-3 hour')) AS sessionStart,
        MAX(DATETIME(dtTransaction, '-3 hour')) AS sessionEnd,
        -- Calculate live minutes using the difference between session start and end times
        (JULIANDAY(MAX(DATETIME(dtTransaction, '-3 hour'))) - 
         JULIANDAY(MIN(DATETIME(dtTransaction, '-3 hour')))) * 24 * 60 AS liveMinutes
    FROM cte_transactions
    GROUP BY idCustomer, sessionDate
),

-- CTE to aggregate live session statistics for each customer
cte_live_stats AS (
    SELECT 
        idCustomer,
        AVG(liveMinutes) AS avgLiveMinutes,
        SUM(liveMinutes) AS totalLiveMinutes,
        MIN(liveMinutes) AS minLiveMinutes,
        MAX(liveMinutes) AS maxLiveMinutes
    FROM cte_live_minutes
    GROUP BY idCustomer
),

-- CTE to calculate lifetime transaction metrics for each customer
cte_lifetime AS (
    SELECT 
        idCustomer,
        COUNT(DISTINCT idTransaction) AS totalTransactionsLifetime,
        -- Average transactions per day since the first transaction
        COUNT(DISTINCT idTransaction) / (MAX(JULIANDAY('{date}') - JULIANDAY(dtTransaction))) AS avgTransactionsPerDay
    FROM transactions
    WHERE dtTransaction < '{date}'
    GROUP BY idCustomer
),

-- CTE to join frequency, live session stats, and lifetime metrics
cte_join AS (
    SELECT 
        f.*,
        l.avgLiveMinutes,
        l.totalLiveMinutes,
        l.minLiveMinutes,
        l.maxLiveMinutes,
        t.totalTransactionsLifetime,
        t.avgTransactionsPerDay
    FROM cte_frequency AS f
    LEFT JOIN cte_live_stats AS l
        ON f.idCustomer = l.idCustomer
    LEFT JOIN cte_lifetime AS t
        ON t.idCustomer = f.idCustomer
)

-- Final SELECT to include the reference date
SELECT 
    '{date}' AS dtRef,
    *
FROM cte_join;
