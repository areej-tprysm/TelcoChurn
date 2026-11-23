# Metric choice

In churn, the two mistakes don't cost the same. If the model misses a customer who is going to leave (false negative), we lose all their future payments. If the model flags a customer who would have stayed (false positive), we only lose the cost of one retention offer. I assumed a missed churner costs 5 and a wasted offer costs 1. For example, missing 100 churners costs 500, but wasting 100 offers costs only 100. Because of this, I chose PR-AUC (average precision) as the primary metric. 
It only looks at the churn class and checks how well the model finds churners without flagging too many customers who would stay. It also works well when the data is imbalanced (only about 26.5% churn), and it does not depend on one fixed threshold. So I use PR-AUC to tune the model, and then I pick the final threshold separately using the cost matrix (5 vs 1).
- Recall is also important as missing a customer who is going to leave is more costly than sending a offer to a customer not going to leave.
## Why not the other metrics

- **Accuracy:** a model that says "nobody churns" gets 0.735 accuracy in my data but catches 0 churners. High accuracy, useless model(e.g: dummy).
- **Recall only:** if the model says "everyone churns", recall becomes 1.0, but we send offers to about 4,140 customers who were never going to leave.
- **Precision only:** if the model flags only the 10 most risky customers, precision can be very high, but hundreds of real churners are missed.
- **F1:** it uses a fixed threshold of 0.5 and treats a missed churner and a wasted offer as equally bad, but for us a missed churner is 5 times worse.
- **ROC-AUC:** it also counts the many easy non-churners, so it can look good even when the model is weak on churners. A random model gets 0.5 in ROC-AUC, but only about 0.265 in PR-AUC, so PR-AUC shows more clearly how good the model really is on churners.