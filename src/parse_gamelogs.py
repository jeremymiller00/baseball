'''Read in and parse lines from a directory of retrosheet data files; prep them for input into Mongodb.

Requires a directory titled "json_files" in the same parent directory as the diretories which contain the retrosheet event files.

Line formats:
{"entry_type": "roster", "player_id": "grimb101", "last_name": "Grimes", "first_name":"Burleigh", "bats": "R", "throws": "R", "team":"BRO", position": "P"}

in general, to import into mongo from command line
mongoimport --jsonArray --db testdb --collection testcoll < test.json

But here is what you should do:
rename all files in dir to prep for mongodb
ls | xargs -I {} mv {} c_{}

to import all in a folder, from inside the folder
ls -1 *.json | sed 's/.json$//' | while read col; do 
    mongoimport --jsonArray --db retrosheet --collection $col < $col.json; 
done

'''
import numpy as np
import sys
import json
import subprocess
import os
import shutil

def read_file(path):
    '''
    Read in a file.

    Parameters:
    ----------
    Input {str}: path to file
    Output: {str}: open python string object
    '''
    with open(path) as f:
        return f.readlines()

def trim_lines(file):
    '''
    Trim '\n' from the end of each line

    Parameters:
    ----------
    Input {str}: open python string object
    Output {str}: open python string object
    '''    
    out = []
    for line in file:
        out.append(line[:-1])
    return out

def split_lines(file):
    '''
    Split string into list of strings; remove quotes

    Parameters:
    ----------
    Input {str}: open python string object
    Output {list: str}: list of event strings
    '''
    out = []
    for line in file:
        cleaned = line.replace("\"", "")
        out.append(cleaned.split(","))
    return out

def build_dicts(file):
    '''
    Convert event string to dictionary with appropriate labels

    Parameters:
    ----------
    Input {list: str}: list of event strings
    Output {list: dict}: list of event dictionaries
    '''
    out = []
    for row in file:

        d = {
            "date":  row[0], "game_num": row[1], "dow": row[2], "visitor": row[3], "visitor_league": row[4], "visitor_game_num": row[5], "home": row[6], "home_league": row[7], "home_game_num": row[8], "visitor_score": row[9], "home_score": row[10], "length_in_outs": row[11], "day_or_night": row[12], "completion_info": row[13], "forfeit_info": row[14], "protest_info": row[15], "park_id": row[16], "attendance": row[17], "time_of_game": row[18], "visitor_line_score": row[19], "home_line_score": row[20], "visitor_ab": row[21], "visitor_hits": row[22], "visitor_doubles": row[23], "visitor_triple": row[24], "visitor_hr": row[25], "visitor_rbi": row[26], "visitor_sac": row[27], "visitor_sacfly": row[28], "visitor_hbp": row[29], "visitor_bb": row[30], "visitor_ibb": row[31], "visitor_k": row[32], "visitor_sb": row[33], "visitor_cs": row[34], "visitor_gdp": row[35], "visitor_ci": row[36], "visitor_lob": row[37], "visitor_pitchers_used": row[38], "visitor_ind_er": row[39], "visitor_team_er": row[40], "visitor_wp": row[41], "visitor_balks": row[42], "visitor_putouts": row[43], "visitor_assists": row[44], "visitor_errors": row[45], "visitor_passed_ball": row[46], "visitor_dp": row[47], "visitor_tp": row[48], "home_ab": row[49], "home_hits": row[50], "home_doubles": row[51], "home_triple": row[52], "home_hr": row[53], "home_rbi": row[54], "home_sac": row[55], "home_sacfly": row[56], "home_hbp": row[57], "home_bb": row[58], "home_ibb": row[59], "home_k": row[60], "home_sb": row[61], "home_cs": row[62], "home_gdp": row[63], "home_ci": row[64], "home_lob": row[65], "home_pitchers_used": row[66], "home_ind_er": row[67], "home_team_er": row[68], "home_wp": row[69], "home_balks": row[70], "home_putouts": row[71], "home_assists": row[72], "home_errors": row[73], "home_passed_ball": row[74], "home_dp": row[75], "home_tp": row[76], "home_ump_id": row[77], "home_ump_name": row[78], "1B_ump_id": row[79], "2B_ump_name": row[80], "2B_ump_id": row[81], "2B_ump_name": row[82], "3B_ump_id": row[83], "3B_ump_name": row[84], "LF_ump_id": row[85], "LF_ump_name": row[86], "RF_ump_id": row[87], "RF_ump_name": row[88], "visitor_manager_id": row[89], "visitor_manager_name": row[90], "home_manager_id": row[91], "home_manager_name": row[92], "winning_pitcher_id": row[93], "winning_pitcher_name": row[94], "losing_pitcher_id": row[95], "losing_pitcher_name": row[96], "saving_pitcher_id": row[97], "saving_pitcher_name": row[98], "gwrbi_batter_id": row[99], "gwrbi_batter_name": row[100], "visitor_starting_pitcher_id": row[101], "visitor_starting_pitcher_name": row[102], "home_starting_pitcher_id": row[103], "home_starting_pitcher_name": row[104], "visitor_starter_1_id": row[105], "visitor_starter_1_name": row[106], "visitor_starter_1_pos": row[107],
            "visitor_starter_2_id": row[108], "visitor_starter_2_name": row[109], "visitor_starter_2_pos": row[110],
            "visitor_starter_3_id": row[111], "visitor_starter_3_name": row[112], "visitor_starter_3_pos": row[113],
            "visitor_starter_4_id": row[114], "visitor_starter_4_name": row[115], "visitor_starter_4_pos": row[116],
            "visitor_starter_5_id": row[117], "visitor_starter_5_name": row[118], "visitor_starter_5_pos": row[119],
            "visitor_starter_6_id": row[120], "visitor_starter_6_name": row[121], "visitor_starter_6_pos": row[122],
            "visitor_starter_7_id": row[123], "visitor_starter_7_name": row[124], "visitor_starter_7_pos": row[125],
            "visitor_starter_8_id": row[126], "visitor_starter_8_name": row[127], "visitor_starter_8_pos": row[128],
            "visitor_starter_9_id": row[129], "visitor_starter_9_name": row[130], "visitor_starter_9_pos": row[131], "home_starter_1_id": row[132], "home_starter_1_name": row[133], "home_starter_1_pos": row[134],
            "home_starter_2_id": row[135], "home_starter_2_name": row[136], "home_starter_2_pos": row[137],
            "home_starter_3_id": row[138], "home_starter_3_name": row[139], "home_starter_3_pos": row[140],
            "home_starter_4_id": row[141], "home_starter_4_name": row[142], "home_starter_4_pos": row[143],
            "home_starter_5_id": row[144], "home_starter_5_name": row[145], "home_starter_5_pos": row[146],
            "home_starter_6_id": row[147], "home_starter_6_name": row[148], "home_starter_6_pos": row[149],
            "home_starter_7_id": row[150], "home_starter_7_name": row[151], "home_starter_7_pos": row[152],
            "home_starter_8_id": row[153], "home_starter_8_name": row[154], "home_starter_8_pos": row[155],
            "home_starter_9_id": row[156], "home_starter_9_name": row[157], "home_starter_9_pos": row[158], "additional_info": row[159], "acquisition_info": row[160]
            }
        
        out.append(d)

    return out

def main(dir):
    '''
    Input: directory with retrosheet files
    Output: folder with json files
    '''
    temp_path = "data/retrosheet_data/json_files"

    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
        print("Directory " , temp_path ,  " Created ")
    else:    
        print("Directory " , temp_path ,  " already exists")

    files = subprocess.check_output(["ls", dir], universal_newlines=True)
    split_files = files.split("\n")
    counter = 0
    for file in split_files:
        if len(file) != 0:
            if file[-3:] == "TXT":
                counter += 1
                print("Processing file {} / {}".format(counter, len(split_files)))
                data = read_file(dir+file)
                trimmed_data = trim_lines(data)
                split_data = split_lines(trimmed_data)
                dicts = build_dicts(split_data)
                outpath = "data/retrosheet_data/json_files/"+file[:-3]+"json"
                with open(outpath, 'w') as fout:
                    json.dump(dicts, fout)
    # try:
    #     shutil.rmtree(temp_path)
    # except OSError as e:
    #     print ("Error: {} - {}}.".format(e.filename, e.strerror))

##################################################################
if __name__ == "__main__":

    '''
    ipython -i src/parse_retrosheet.py directory
    '''
    main(sys.argv[1])


