'''
Author: your name
Date: 2021-04-29 20:14:02
LastEditTime: 2021-05-10 15:10:56
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /code_for_ore/one_attempt.py
'''

from train_main_model import main
from test import test
from config import FLAGS

best_f1 = 0.0
for i in range(5):
    FLAGS.knowledges = ["kbRel"]
    with open("out/super.txt", "a") as logf:
        logf.write(str(FLAGS.knowledges) + "\n")
    main(best_f1)
    f1 = test()
    best_f1 = max(f1, best_f1)
# FLAGS.knowledges = ["desc"]
# with open("out/super.txt", "a") as logf:
#     logf.write(str(FLAGS.knowledges) + "\n")
# main()
# test()

# FLAGS.knowledges = ["exrest"]
# with open("out/super.txt", "a") as logf:
#     logf.write(str(FLAGS.knowledges) + "\n")
# main()
# test()
# for i in range(5):
#     FLAGS.knowledges = ["kbRel"]
#     with open("out/super.txt", "a") as logf:
#         logf.write(str(FLAGS.knowledges) + "\n")
#     main()
#     test()

# FLAGS.knowledges = ["desc", "exrest"]
# with open("out/super.txt", "a") as logf:
#     logf.write(str(FLAGS.knowledges) + "\n")
# main()
# test()

# FLAGS.knowledges = ["desc", "kbRel"]
# with open("out/super.txt", "a") as logf:
#     logf.write(str(FLAGS.knowledges) + "\n")
# main()
# test()

# FLAGS.knowledges = ["exrest", "kbRel"]
# with open("out/super.txt", "a") as logf:
#     logf.write(str(FLAGS.knowledges) + "\n")
# main()
# test()

# FLAGS.knowledges = ["desc", "exrest", "kbRel"]
# with open("out/super.txt", "a") as logf:
#     logf.write(str(FLAGS.knowledges) + "\n")
# main()
# test()
