# NBA-team-clustering

Files relevant to visualizations and balance score calculations in https://fansided.com/2017/06/01/nylon-calculus-nba-finals-clustering-team-offensive-styles/

Please credit Senthil S. Natarajan if using or modifying the work contained herein, or contact via Twitter @SENTH1S

Files Overview:
---
1. **"team_offenses.csv":** CSV file containing data on various offensive style attributes, pulled from NBA Stats

2. **"team_offenses_percentilever.csv":** CSV version of percentile converted offensive data for each team

3. **"adv_201617gamelogs.csv":** CSV file containing all game logs from 2016-17 NBA Season (via BBall-Ref)

4. **"team_offense_styles.py":** main python script for clustering NBA teams using KMeans clustering and returning visualization with each team tagged to a cluster

5. **"team_radar.py":** main python script for getting requisite radar charts for various teams to visualize their styles

6. **"radar.py":** python script for drawing radar charts

7. **"defenseagainst.py":** quickly calculate how each team performed defensively versus each type of offensive style

8. **"clustertags.csv":** CSV file containing transcribed cluster tags based on results of team_offense_styles.py
