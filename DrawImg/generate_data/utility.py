import pandas as pd

def name_dict():
    name_dict = {
        "100522-501": "051122-501",
        "131122-401": "131222-401",
        "141122-401": "141222-401",
        "141122-402": "141222-402",
        "200522-501": "311022-501",
    }

    info = pd.read_excel("./PP_info_all.xlsx")

    session1 = info["subj_id"]
    session2 = info["cubric_id"]
    session2 = [name_dict[s] if s in name_dict.keys() else s for s in session2]

    dict_1_2 = {}
    dict_2_1 = {}
    for i in range(len(session1)):
        if isinstance(session2[i], float):
            continue
        dict_1_2.update({session1[i]: session2[i]})
        dict_2_1.update({session2[i]: session1[i]})

    return dict_1_2, dict_2_1