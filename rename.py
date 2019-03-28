import os


class BatchRename():


    def __init__(self):
        self.path = 'D:\\BehaviorAnalysis\\trunk\\src\\poseEstimation1.0.1\\data\\temp_image2\\img\\Relax'

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 1
        for item in filelist:
            if item.endswith('.jpg'):
	    
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), "R-elax"+"_6"+str(i) + '.jpg')
                try:
                    os.rename(src, dst)
                    print ('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    continue
        print ('total %d to rename & converted %d jpgs' % (total_num, i))

if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()

"""


class BatchRename():

    def __init__(self):
        self.path = '/home/bz/PycharmProjects/Bozhon/1'

    def rename(self):
        #i = 1000000
        for dirs in os.listdir(self.path):

            #print(dirs)
            for  file in os.listdir(self.path + '/' + dirs):
                jpgfile = os.path.join(self.path, dirs, file)
                #print jpgfile
                #if jpgfile.endswith('.jpg'):

                    #src = os.path.join(os.path.abspath(self.path), jpgfile)
                dst = os.path.join(self.path, str(i) +dirs+ '.jpg')
                print dst
                    #try:
                        #os.rename(jpgfile, dst)
                        #print 'converting %s to %s ...' % (src, dst)
                        #i = i + 1
                    #except:
                        #continue
            #print  i


if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()


"""
