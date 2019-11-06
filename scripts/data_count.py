import copy
def filename_format(f_type, onedata, lang):
    return onedata+"."+lang

def data_count(dataset, src, tgt, f_type):
    ret_dict = {}
    for onedata in dataset:
        data_dict = {"count":0, src+"-total-len":0, tgt+"-total-len":0, src+"-maxlen":0, tgt+"-maxlen":0}
        with open(filename_format(f_type, onedata, src), "r") as src_reader, open(filename_format(f_type, onedata, tgt), "r") as tgt_reader:
            for s_data, t_data in zip(src_reader, tgt_reader):
                data_dict["count"] += 1
                s_data, t_data = s_data.strip(), t_data.strip()
                s_len, t_len = len(s_data.split()), len(t_data.split())
                data_dict[src+"-total-len"] += s_len
                data_dict[tgt+"-total-len"] += t_len
                data_dict[src+"-maxlen"] = max(data_dict[src+"-maxlen"], s_len)
                data_dict[tgt+"-maxlen"] = max(data_dict[tgt+"-maxlen"], t_len)
        data_dict[src+"-avglen"] = data_dict[src+"-total-len"]/data_dict["count"]
        data_dict[tgt+"-avglen"] = data_dict[tgt+"-total-len"]/data_dict["count"]
        ret_dict[onedata] = data_dict
    return ret_dict

if __name__ == "__main__":
    src = "art"
    tgt = "abs"

    dataset = ["train", "valid", "test"]
    ret_dict = copy.deepcopy(data_count(dataset, src, tgt, "cnndm"))
    print(ret_dict)

