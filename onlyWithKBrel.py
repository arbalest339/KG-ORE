import json

rf = open("coerkb/test.txt", "r")
wf = open("coerkb/testWithRel.txt", "w")
for line in rf:
    jline = json.loads(line)
    if jline["kbRel"]:
        wf.write(line)