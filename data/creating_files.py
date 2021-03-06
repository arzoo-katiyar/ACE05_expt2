sentences_file = open("sentence_id_ACE05_Coref.txt")

file_sent_dict = dict()

sentid = -1

for sentence in sentences_file.readlines():
    fileid = sentence.split("\t")[0]
    #sentid += int(sentence.split("\t")[1])
    sentid += 1
    filename = sentence.strip().split("\t")[2]
    
    if sentid not in file_sent_dict.keys():
        file_sent_dict[sentid] = filename

#print(file_sent_dict)


train_file = open("filelist_ACE05/filelist_train.lst")
dev_file = open("filelist_ACE05/filelist_dev.lst")
test_file = open("filelist_ACE05/filelist_test.lst")


train_lst = []
dev_lst = []
test_lst = []

for filename in train_file.readlines():
    train_lst.append(filename.strip())

for filename in dev_file.readlines():
    dev_lst.append(filename.strip())

for filename in test_file.readlines():
    test_lst.append(filename.strip())


total_files = 0
    
for sentid in file_sent_dict.keys():
    if file_sent_dict[sentid] in train_lst:
        total_files += 1

#print(len(test_lst))
#print(total_files)
#print(len(file_sent_dict.keys()))
'''exit()'''
    

train_file = open("train.txt", "w+")
dev_file = open("dev.txt", "w+")
test_file = open("test.txt", "w+")

labels_file = open("anno_ACE05_BILOU_POS_Coref_serial.txt")

sentid = -1
curr_sent = ""
previous_filename = ""

files_count = 0

for token in labels_file.readlines():
    if len(token.split()) < 2:
        sentid += 1
        #file_sent_dict
        filename = file_sent_dict[sentid]

        if filename != previous_filename:
            
            #files_count += 1
            previous_filename = filename
            if filename in train_lst:
                train_file.write("-DOCSTART-\tO\tnull\t0\n")
            elif filename in dev_lst:
                dev_file.write("-DOCSTART-\tO\tnull\t0\n")
            elif filename in test_lst:
                #files_count += 1
                test_file.write("-DOCSTART-\tO\tnull\t0\n")
            else:
                print("Does not belong to any split!!!!")
                exit()

        
        if filename in train_lst:
            train_file.write((curr_sent.replace("U_", "S-")).replace("L_", "E-").replace("I_", "I-").replace("B_", "B-"))
            train_file.write("\n")
        elif filename in dev_lst:
            dev_file.write((curr_sent.replace("U_", "S-")).replace("L_", "E-").replace("I_", "I-").replace("B_", "B-"))
            dev_file.write("\n")
        elif filename in test_lst:
            files_count += 1
            #print(curr_sent)
            #print("====")
            test_file.write((curr_sent.replace("U_", "S-")).replace("L_", "E-").replace("I_", "I-").replace("B_", "B-"))
            test_file.write("\n")
        else:
            print("Does not belong to any split!!!!")
            exit()

        curr_sent = ""

    else:
        curr_sent += token
        
train_file.close()
dev_file.close()
test_file.close()

print(files_count)
print(sentid)
