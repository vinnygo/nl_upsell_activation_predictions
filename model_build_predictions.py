# creating counterfactual control group
import numpy as np
import pandas as pd
import pandas_gbq
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# training: 2020-03-16 - 2022-02-28 (or week 13 of 2020 onwards)
# validating: 2022-03-01 through 2022-04-13 sub starts
# out of sample prediction: 2022-04-14 onwards
dates = pd.date_range(start='2022-03-01', end='2022-04-13')
sub_dates = pd.date_range(start='2020-03-16', end='2022-02-28')

training_df_query = """
WITH subs as (
  select 
    s.sub_type as entitlement_type, 
    s.regi_id, 
    date(datetime(s.subscriber_start_ts, "America/New_York")) as subscriber_start_dt,
    r.sub_type,
    r.billing_frequency
  from (select * from `nyt-bigquery-beta-workspace.nytcooking.subs_all` where sub_type in ('ck' , 'ada') and date(datetime(subscriber_start_ts, "America/New_York")) between '2020-03-16' and '2022-02-28') s
  join `nyt-bigquery-beta-workspace.nytcooking.retention_base` r
  on s.subscriber_id = r.subscriber_id
  where r.sub_type = 'iOS'
  ),
-- at weekly level
visits as (
  select
    s.entitlement_type,
    s.sub_type, 
    s.billing_frequency,
    p.regi_id,
    s.subscriber_start_dt,
    count(distinct case when ceil(date_diff(p.date, s.subscriber_start_dt, day) / 7) = 0 then 1 else ceil(date_diff(p.date, s.subscriber_start_dt, day) / 7) end) as num_wks_visit
  from subs s
  left join `nyt-bigquery-beta-workspace.nytcooking.page` p
  on cast(s.regi_id as string) = cast(p.regi_id as string) 
  and date(p._partitiontime) >= '2020-01-01'
  and date_diff(date(p._partitiontime), s.subscriber_start_dt, day) between 0 and 27
  group by 1, 2, 3, 4, 5
),
wk_agg as (
  select 
    v.entitlement_type,
    v.sub_type, 
    v.billing_frequency,
    v.regi_id,
    v.subscriber_start_dt,
    b.save,
    b.ERP,
    b.crossplatform,
    --date_trunc(subscriber_start_dt, week(monday)) as week, 
    case when v.num_wks_visit = 4 then 1 else 0 end as wau_at_day28
from visits v
join nytcooking.behavior_lookback_28day b
on v.regi_id = b.regi_id
and date_diff(date(b._partitiontime), v.subscriber_start_dt, day) = 28
), 
pre_results as (
select 
  --week,
  subscriber_start_dt,
  avg(case when save >= 1 then 1 else 0 end) as pct_save_in_mo1,
  avg(case when ERP >= 1 then 1 else 0 end) as pct_ERP_in_mo1,
  avg(crossplatform) as pct_crossplatform_in_mo1,
  avg(case when entitlement_type = 'ck' then 1 else 0 end) pct_ck_start, 
  avg(case when billing_frequency = 'monthly' then 1 else 0 end) as pct_monthly_start,
  avg(wau_at_day28) as pct_wau
from wk_agg
group by 1
)
select 
  extract(year from subscriber_start_dt) as yr,
  extract(week from subscriber_start_dt) as wk_of_yr,
  extract(DAYOFWEEK FROM subscriber_start_dt) as day_of_wk,
  subscriber_start_dt,
  --week, 
  pct_save_in_mo1,
  pct_ERP_in_mo1,
  pct_crossplatform_in_mo1,
  pct_ck_start,
  pct_monthly_start,
  pct_wau
from pre_results

order by 1, 2, 3
"""
test_df_query = """
WITH subs as (
  select 
    s.sub_type as entitlement_type, 
    s.regi_id, 
    date(datetime(s.subscriber_start_ts, "America/New_York")) as subscriber_start_dt,
    r.sub_type,
    r.billing_frequency
  from (select * from `nyt-bigquery-beta-workspace.nytcooking.subs_all` where sub_type in ('ck' , 'ada') and date(datetime(subscriber_start_ts, "America/New_York")) between '2022-03-01' and '2022-04-13') s
  join `nyt-bigquery-beta-workspace.nytcooking.retention_base` r
  on s.subscriber_id = r.subscriber_id
  where r.sub_type = 'iOS'
  ),
-- at weekly level
visits as (
  select
    s.entitlement_type,
    s.sub_type, 
    s.billing_frequency,
    p.regi_id,
    s.subscriber_start_dt,
    count(distinct case when ceil(date_diff(p.date, s.subscriber_start_dt, day) / 7) = 0 then 1 else ceil(date_diff(p.date, s.subscriber_start_dt, day) / 7) end) as num_wks_visit
  from subs s
  left join `nyt-bigquery-beta-workspace.nytcooking.page` p
  on cast(s.regi_id as string) = cast(p.regi_id as string) 
  and date(p._partitiontime) >= '2019-01-01'
  and date_diff(date(p._partitiontime), s.subscriber_start_dt, day) between 0 and 27
  group by 1, 2, 3, 4, 5
),
wk_agg as (
  select 
    v.entitlement_type,
    v.sub_type, 
    v.billing_frequency,
    v.regi_id,
    v.subscriber_start_dt,
    b.save,
    b.ERP,
    b.crossplatform,
    --date_trunc(subscriber_start_dt, week(monday)) as week, 
    case when v.num_wks_visit = 4 then 1 else 0 end as wau_at_day28
from visits v
join nytcooking.behavior_lookback_28day b
on v.regi_id = b.regi_id
and date_diff(date(b._partitiontime), v.subscriber_start_dt, day) = 28
), 
pre_results as (
select 
  --week,
  subscriber_start_dt,
  avg(case when save >= 1 then 1 else 0 end) as pct_save_in_mo1,
  avg(case when ERP >= 1 then 1 else 0 end) as pct_ERP_in_mo1,
  avg(crossplatform) as pct_crossplatform_in_mo1,
  avg(case when entitlement_type = 'ck' then 1 else 0 end) pct_ck_start, 
  avg(case when billing_frequency = 'monthly' then 1 else 0 end) as pct_monthly_start,
  avg(wau_at_day28) as pct_wau
from wk_agg
group by 1
)
select 
  extract(year from subscriber_start_dt) as yr,
  extract(week from subscriber_start_dt) as wk_of_yr,
  extract(DAYOFWEEK FROM subscriber_start_dt) as day_of_wk,
    subscriber_start_dt,

  --week, 
  pct_save_in_mo1,
  pct_ERP_in_mo1,
  pct_crossplatform_in_mo1,
  pct_ck_start,
  pct_monthly_start,
  pct_wau
from pre_results
order by 1, 2, 3
"""
out_of_sample_prediction_query = """
WITH subs as (
  select 
    s.sub_type as entitlement_type, 
    s.regi_id, 
    date(datetime(s.subscriber_start_ts, "America/New_York")) as subscriber_start_dt,
    r.sub_type,
    r.billing_frequency
  from (select * from `nyt-bigquery-beta-workspace.nytcooking.subs_all` where sub_type in ('ck' , 'ada') and date(datetime(subscriber_start_ts, "America/New_York")) >= '2022-04-14') s
  join `nyt-bigquery-beta-workspace.nytcooking.retention_base` r
  on s.subscriber_id = r.subscriber_id
  where r.sub_type = 'iOS'
  ),
-- at weekly level
visits as (
  select
    s.entitlement_type,
    s.sub_type, 
    s.billing_frequency,
    p.regi_id,
    s.subscriber_start_dt,
    count(distinct case when ceil(date_diff(p.date, s.subscriber_start_dt, day) / 7) = 0 then 1 else ceil(date_diff(p.date, s.subscriber_start_dt, day) / 7) end) as num_wks_visit
  from subs s
  left join `nyt-bigquery-beta-workspace.nytcooking.page` p
  on cast(s.regi_id as string) = cast(p.regi_id as string) 
  and date(p._partitiontime) >= '2019-01-01'
  and date_diff(date(p._partitiontime), s.subscriber_start_dt, day) between 0 and 27
  group by 1, 2, 3, 4, 5
),
wk_agg as (
  select 
    v.entitlement_type,
    v.sub_type, 
    v.billing_frequency,
    v.regi_id,
    v.subscriber_start_dt,
    b.save,
    b.ERP,
    b.crossplatform,
    --date_trunc(subscriber_start_dt, week(monday)) as week, 
    case when v.num_wks_visit = 4 then 1 else 0 end as wau_at_day28
from visits v
join nytcooking.behavior_lookback_28day b
on v.regi_id = b.regi_id
and date_diff(date(b._partitiontime), v.subscriber_start_dt, day) = 28
), 
pre_results as (
select 
  --week,
  subscriber_start_dt,
  avg(case when save >= 1 then 1 else 0 end) as pct_save_in_mo1,
  avg(case when ERP >= 1 then 1 else 0 end) as pct_ERP_in_mo1,
  avg(crossplatform) as pct_crossplatform_in_mo1,
  avg(case when entitlement_type = 'ck' then 1 else 0 end) pct_ck_start, 
  avg(case when billing_frequency = 'monthly' then 1 else 0 end) as pct_monthly_start,
  avg(wau_at_day28) as pct_wau
from wk_agg
group by 1
)
select 
  extract(year from subscriber_start_dt) as yr,
  extract(week from subscriber_start_dt) as wk_of_yr,
  extract(DAYOFWEEK FROM subscriber_start_dt) as day_of_wk,
    subscriber_start_dt,

  --week, 
  pct_save_in_mo1,
  pct_ERP_in_mo1,
  pct_crossplatform_in_mo1,
  pct_ck_start,
  pct_monthly_start,
  pct_wau
from pre_results
order by 1, 2, 3
"""

training_df = pandas_gbq.read_gbq(training_df_query, project_id = 'nyt-bigquery-beta-workspace')
test_df = pandas_gbq.read_gbq(test_df_query, project_id = 'nyt-bigquery-beta-workspace')
df = pd.concat([training_df, test_df])
df = pd.get_dummies(df, columns=['day_of_wk', 'wk_of_yr', 'yr'])

sub_training_Y = df[df['subscriber_start_dt'] <= '2022-02-28']['pct_wau']
sub_training_X = df[df['subscriber_start_dt'] <= '2022-02-28'].drop('pct_wau',axis=1).drop('subscriber_start_dt',axis=1)
test_Y = df[df['subscriber_start_dt'] >= '2022-03-01']['pct_wau']
test_X = df[df['subscriber_start_dt'] >= '2022-03-01'].drop('pct_wau',axis=1).drop('subscriber_start_dt',axis=1)

# RANDOM FOREST
params = {
    'min_samples_split':[2, 4, 6, 8, 10],
    'max_features':[0.4, 0.45, 0.5, 0.55, 0.6],
    'min_samples_leaf':[0.01, 0.015, 0.02, 0.025, 0.03],
    'ccp_alpha': np.linspace(start=0.0, stop=1.0, num=20)
}
rf = RandomForestRegressor(n_estimators=500, bootstrap=True, random_state=10419491)
clf = GridSearchCV(rf, param_grid=params, verbose=5, cv=5, n_jobs=-1)
clf.fit(sub_training_X, sub_training_Y)

# included this block after deciding to include pruning complexity paraemter for more regularization
best_rf_ccp = RandomForestRegressor(
    **clf.best_params_
)
best_rf_ccp.fit(sub_training_X, sub_training_Y)
rf_ccp_pred = best_rf_ccp.predict(test_X)
rf_ccp_insample_pred = best_rf_ccp.predict(sub_training_X)
rf_ccp_rmse = mean_squared_error(rf_ccp_pred, test_Y, squared=False)

# best_rf = RandomForestRegressor(
#     **clf.best_params_
# )
# best_rf.fit(sub_training_X, sub_training_Y)
# 
# #prediction
# rf_pred = best_rf.predict(test_X)
# rf_insample_pred = best_rf.predict(sub_training_X)
# rf_rmse = mean_squared_error(rf_pred, test_Y, squared=False)


# LASSO  
from sklearn.linear_model import LassoCV
linreg = LassoCV(cv=10, random_state=10419491, alphas=np.linspace(start=0.0001, stop=10, num=10000))
linreg.fit(sub_training_X, sub_training_Y)
lr_sub_pred = linreg.predict(test_X)
lr_insample_pred = linreg.predict(sub_training_X)

# RIDGE
from sklearn.linear_model import RidgeCV
ridge = RidgeCV(alphas=np.linspace(start=0.0001, stop=10, num=10000), cv=10)
ridge.fit(sub_training_X, sub_training_Y)
ridge_sub_pred = ridge.predict(test_X)
ridge_insample_pred = ridge.predict(sub_training_X)



# OUT OF SAMPLE PREDICTIONS PLOT
pred_df = pd.DataFrame({'sub_start_dt': dates, 'actual': test_Y, 'rf_pred':rf_ccp_pred, 'lasso_pred': lr_sub_pred, 'ridge_pred': ridge_sub_pred})
pred_df = pd.melt(pred_df, id_vars=['sub_start_dt'], value_vars=['actual','rf_pred', 'lasso_pred', 'ridge_pred'])
pred_df['value'] = pred_df['value']*100
pred_df.columns = ['Date', 'Model', 'Activation Rate (%)']
sns.lineplot(data=pred_df, x="Date", y='Activation Rate (%)', hue='Model')
#pred_df['week'] = pred_df['sub_start_dt'].dt.to_period('W').apply(lambda r: r.start_time)
#pred_df = pred_df.groupby(['week', 'variable'])['value'].mean().reset_index()
#sns.lineplot(data=pred_df, x="week", y='value', hue='variable')

#checking the # of pct points difference
pred_df = pd.DataFrame({'sub_start_dt': dates, 'actual': test_Y, 'rf_pred':rf_ccp_pred, 'lasso_pred': lr_sub_pred, 'ridge_pred': ridge_sub_pred})
pred_df['rf_diff'] = pred_df['rf_pred'] - pred_df['actual']
pred_df['lasso_diff'] = pred_df['lasso_pred'] - pred_df['actual']
pred_df['ridge_diff'] = pred_df['ridge_pred'] - pred_df['actual']
pred_df = pd.melt(pred_df, id_vars=['sub_start_dt'], value_vars=['rf_diff','lasso_diff', 'ridge_diff'])
pred_df['value'] = pred_df['value'] * 100
pred_df.columns = ['Date', 'Model', 'Percentage Point Difference Over Actual Activation Rate']
g = sns.lineplot(data=pred_df, x="Date", y='Percentage Point Difference Over Actual Activation Rate', hue='Model')
ax = g.axes
ax.axhline(0.0, ls="--", color="red")

# IN SAMPLE PREDICTIONS PLOT
pred_df = pd.DataFrame({
    'sub_start_dt': sub_dates,
    'actual': sub_training_Y,
    'rf_pred': rf_ccp_insample_pred,
    'lasso_pred': lr_insample_pred,
    'ridge_pred': ridge_insample_pred
})
pred_df = pd.melt(pred_df, id_vars=['sub_start_dt'], value_vars=['actual','rf_pred', 'lasso_pred', 'ridge_pred'])
#sns.lineplot(data=pred_df, x="sub_start_dt", y='value', hue='variable')
# get to weekly averages
pred_df['week'] = pred_df['sub_start_dt'].dt.to_period('W').apply(lambda r: r.start_time)
pred_df = pred_df.groupby(['week', 'variable'])['value'].mean().reset_index()
pred_df['value'] = pred_df['value']*100
pred_df.columns = ['Week', 'Model', 'Activation Rate (%)']
sns.lineplot(data=pred_df, x="Week", y='Activation Rate (%)', hue='Model')

# RMSEs
lasso_rmse = mean_squared_error(test_Y, lr_sub_pred, squared=False)
ridge_rmse = mean_squared_error(test_Y, ridge_sub_pred, squared=False)

rf_rmse
rf_ccp_rmse
lasso_rmse
ridge_rmse
(rf_ccp_rmse/rf_rmse )- 1

# looking at % differences in RMSE in test period vs training period
(rf_ccp_rmse/mean_squared_error(sub_training_Y, rf_ccp_insample_pred, squared=False)) - 1
(lasso_rmse/mean_squared_error(sub_training_Y, lr_insample_pred, squared=False))-1
(ridge_rmse/mean_squared_error(sub_training_Y, ridge_insample_pred, squared=False))-1

##### PREDICTIONS AFTER SELECTING RF MODEL ###########
# making predictions for April 14th onward!!
pred_pd_df = pandas_gbq.read_gbq(out_of_sample_prediction_query, project_id = 'nyt-bigquery-beta-workspace')

pred_pd_df = pd.get_dummies(pred_pd_df, columns=['day_of_wk', 'wk_of_yr', 'yr'])
pred_pd_Y = pred_pd_df['pct_wau']
pred_pd_X = pred_pd_df.drop('pct_wau',axis=1).drop('subscriber_start_dt',axis=1)

#not all features in here because limited time frame, so doesnt have all time dummy variables from model
zero_fill = pd.DataFrame(0, index=np.arange(len(pred_pd_X)), columns=test_X.columns)
pred_pd_X_filled = pd.concat([pred_pd_X, zero_fill[zero_fill.columns.difference(pred_pd_X.columns)]], axis=1)

pred_pd_rf_predictions = best_rf_ccp.predict(pred_pd_X_filled)

prediction_df = pd.DataFrame({
    'sub_start_dt': pred_pd_df['subscriber_start_dt'],
    'activation_rate': pred_pd_rf_predictions
})
pandas_gbq.to_gbq(prediction_df, destination_table=f"gonzales_vincent.nl_upsell_activation_daily_predictions", project_id='nyt-bigquery-beta-workspace', if_exists='replace')
