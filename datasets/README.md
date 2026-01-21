## Dataset format describe

* date: date of map
* team_A, team_B: names of teams
* map: name of map
* bo_type: best of type match
* lan: 0 - online, 1 - lan
* odds_A, odds_B: coeffs from bookmaker
* hltv_points_diff: hltv_points_diff = hltv_points_A - hltv_points_B
* valve_points_diff: valve_points_diff = valve_points_A - valve_points_B
* form_A,form_B: winrate team last 3 mounth
* ln_games_A,ln_games_B: ln_games_A = ln(games_A)
* map_wr_A,map_wr_B: winrate team on map last 3 mounth
* h2h_A: h2h_A = (wins_A_vs_b + 1) / (maps_A_vs_b + 2)
* avg_rating_3_0_diff: difference between mean rating 3.0 of team_A and team_B
* avg_rating_diff: avg_rating_diff = alpha * avg_rating_3_0_A - beta * avg_rating_3_0_A; 
    if team_A/B in tier-1, alpha/beta = 1.0

    if team_A/B in tier-1.5, alpha/beta = 0.9; 

    if team_A/B in tier-2, alpha/beta = 0.75; 

    if team_A/B in tier-3, alpha/beta = 0.6; 

    tier-1 - top 10, tier-1.5 - top 20, tier-2 - top 50 from valve ranking
* coach_wr_A,coach_wr_B: coache's winrate last 3 mounth
* pick_a_winner_A: probably of win team_A according to the opinion HLTV
* roster_changed_last_30d_A,roster_changed_last_30d_B: quantity of changes players in team
* winner_A: 0 - team_A lose, 1 - team_A win