WITH tb_fl_churn AS (
    SELECT 
        t1.dtRef,
        t1.idCustomer,
        CASE WHEN t2.idCustomer IS NULL THEN 1 ELSE 0 END AS flChurn
    FROM 
        fs_general AS t1

    -- Join to check if a record exists for the same customer 21 days in the future
    LEFT JOIN 
        fs_general AS t2 
        ON t2.idCustomer = t1.idCustomer
        AND t1.dtRef = DATE(t2.dtRef, '-21 day')

    -- Filter records where the reference date is at least 21 days before '2024-06-06'
    WHERE (t1.dtRef < DATE('2024-06-07', '-21 day')
    AND strftime('%d', t1.dtRef) = '01')
    OR t1.dtRef = DATE('2024-06-07', '-21 day') -- creating the OOT, the last mature photo of our active base
)

-- Creating the Analytical Base Table (ABT)
SELECT 
    t1.dtRef,
    t1.idCustomer,
    t1.flChurn,

    -- Features from fs_general (t2)
    t2.recencyDays,
    t2.frequencyDays,
    t2.pointsValue,
    t2.baseAgeDays,
    t2.flEmail,

    -- Features from fs_hour (t3)
    t3.pointsMorning,
    t3.pointsAfternoon,
    t3.pointsEvening,
    t3.pctPointsMorning,
    t3.pctPointsAfternoon,
    t3.pctPointsEvening,
    t3.transactionsMorning,
    t3.transactionsAfternoon,
    t3.transactionsEvening,
    t3.pctTransactionsMorning,
    t3.pctTransactionsAfternoon,
    t3.pctTransactionsEvening,

    -- Features from fs_points (t4)
    t4.balancePontsW3,
    t4.balancePointsW2,
    t4.balancePointsW1,
    t4.pointsCumulativeW3,
    t4.pointsCumulativeW2,
    t4.pointsCumulativeW1,
    t4.pointsRedeemW3,
    t4.pointsRedeemW2,
    t4.pointsRedeemW1,
    t4.pointsBalance,
    t4.pointsCumulativeLife,
    t4.pointsRedeemLife,
    t4.pointsPerDay,

    -- Features from fs_products (t5)
    t5.qtyChatMessage,
    t5.qtyAttendanceList,
    t5.qtyRedeemPony,
    t5.qtyPointsExchange,
    t5.qtyStreakPresence,
    t5.qtyAirflowLover,
    t5.qtyRLover,
    t5.pointsChatMessage,
    t5.pointsAttendanceList,
    t5.pointsRedeemPony,
    t5.pointsPointsExchange,
    t5.pointsStreakPresence,
    t5.pointsAirflowLover,
    t5.pointsRLover,
    t5.pctChatMessage,
    t5.pctAttendanceList,
    t5.pctRedeemPony,
    t5.pctPointsExchange,
    t5.pctStreakPresence,
    t5.pctAirflowLover,
    t5.pctRLover,
    t5.avgChatLive,
    t5.maxQuantityProduct,

    -- Features from fs_sessions (t6)
    t6.daysCountD21,
    t6.daysCountD14,
    t6.daysCountD7,
    t6.avgLiveMinutes,
    t6.totalLiveMinutes,
    t6.minLiveMinutes,
    t6.maxLiveMinutes,
    t6.totalTransactionsLifetime,
    t6.avgTransactionsPerDay

FROM 
    tb_fl_churn AS t1

-- Join with fs_general (t2)
LEFT JOIN 
    fs_general AS t2
    ON t1.idCustomer = t2.idCustomer
    AND t1.dtRef = t2.dtRef

-- Join with fs_hour (t3) for session and transaction metrics
LEFT JOIN 
    fs_hour AS t3
    ON t1.idCustomer = t3.idCustomer
    AND t1.dtRef = t3.dtRef

-- Join with fs_points (t4) for cumulative and balance metrics
LEFT JOIN 
    fs_points AS t4
    ON t1.idCustomer = t4.idCustomer
    AND t1.dtRef = t4.dtRef

-- Join with fs_products (t5) for product-related metrics
LEFT JOIN 
    fs_products AS t5
    ON t1.idCustomer = t5.idCustomer
    AND t1.dtRef = t5.dtRef

-- Join with fs_sessions (t6) for live session metrics
LEFT JOIN 
    fs_transactions AS t6
    ON t1.idCustomer = t6.idCustomer
    AND t1.dtRef = t6.dtRef

ORDER BY 
    t1.dtRef, t1.idCustomer;
