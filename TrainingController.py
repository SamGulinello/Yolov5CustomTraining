import os
import shutil
import yaml
import cv2

# Shell command requires the items to be in the form of a string
def listToString(inpt):

    outputString = ""
    for i in inpt:
        outputString = outputString + i + " "

    return outputString
    
# Create list of wanted items for data set
def getItems(fileName):

    with open(fileName) as objectFile:
        objectList = []

        for i in objectFile:
            objectList.append(i.rstrip())

    return(objectList)

def formatFileStruct(items):

    print('Formatting files for Yolov5')
    classList = ['test', 'train', 'validation']
    dataSet = 'iSeek_Objects'
    for i in classList:
        folderPath = './{}/labels/{}'.format(dataSet, i + '/')
        if not os.path.isdir(folderPath):
            os.makedirs(folderPath)

        for t in items:
            labelPath = './OID/Dataset/{}/{}/Label/'.format(i,t)
            fileNames = os.listdir(labelPath)

            for fileName in fileNames:
                with open(os.path.join(labelPath, fileName)) as f:
                    data = f.readlines()

                nd = []
                for line in data:
                    item = line.split()[0]
                    index = items.index(item)

                    # OIDv4_Toolkit gives us these values.
                    leftX = float(line.split()[1])
                    topY = float(line.split()[2])
                    rightX = float(line.split()[3])
                    bottomY = float(line.split()[4])

                    # yolov5 requires these values but we need to scale them where xMax and yMax are both 1
                    xMiddle = (leftX + rightX) / 2
                    yMiddle = (topY + bottomY) / 2
                    boxWidth = rightX - leftX
                    boxHeight = (bottomY - topY)
                    
                    # To scale the coordinates we must have the image size
                    filePath = './OID/Dataset/{}/{}/'.format(i, t) + fileName.replace('.txt', '.jpg')
                    print(filePath)
                    img = cv2.imread(filePath)
                    imgWidth, imgHeight = img.shape[:2]

                    # These are the values yolov5 needs at the correct scale
                    xMiddle = xMiddle / imgWidth
                    boxWidth = boxWidth / imgWidth
                    yMiddle = yMiddle / imgHeight
                    boxHeight = boxHeight / imgHeight

                    newLine = "{} {} {} {} {}\n".format(str(index), str(xMiddle), str(yMiddle), str(boxWidth), str(boxHeight))
                    print(newLine)
                    nd.append(newLine)
                    
                with open(os.path.join(labelPath, fileName), 'w') as f:
                    f.writelines(nd)
     
                shutil.move(os.path.join(labelPath, fileName), folderPath)

        folderPath = './{}/images/{}'.format(dataSet, i + '/')
        if not os.path.isdir(folderPath):
            os.makedirs(folderPath)
        
        for t in items:
            imagePath = './OID/Dataset/{}/{}/'.format(i,t)
            fileNames = os.listdir(imagePath)
            fileNames = fileNames[:-1]
            
            for fileName in fileNames:
                shutil.move(os.path.join(imagePath, fileName), folderPath)

    print('Done Formatting')

def main():
    # Download Images using OIDv4
    itemsList = getItems("testItems.txt")
    commandString = "python3 OIDv4_ToolKit/main.py downloader -y --classes {}--type_csv all --limit 100".format(listToString(itemsList))
    os.system(commandString)
    formatFileStruct(itemsList)

    nc = len(itemsList)
    dictFile = {'train' : './iSeek_Objects/images/train', 'test' : './iSeek_Objects/images/test', 'val' : './iSeek_Objects/images/validation', 'nc': nc,'names': itemsList}

    with open(r'./data/test.yaml', 'w') as file:
        documents = yaml.dump(dictFile, file)

    # Execute Trainig using Yolov5
    commandString = "python3 train.py --data test.yaml --weights yolov5s.pt --cfg yolov5s.yaml --epochs 5 --batch 1 --workers 1"
    os.system(commandString)

main()

