import numpy as np
import cv2

# Let's take a look at our digits dataset
file = 'D:/bikram/machine learning/PracticeSets/sudoku/sudoku.png'
image = cv2.imread(file)
ret,thresh1 = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
cv2.imwrite('D:/bikram/machine learning/PracticeSets/sudoku/BinarySudoku.png',thresh1)
file = 'D:/bikram/machine learning/PracticeSets/sudoku/BinarySudoku.png'
image_slicer.slice(file, 9*9)
image = cv2.imread(file)
#gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#cv2.imshow('large image', gray)
#small = cv2.pyrDown(image)
#cv2.imshow('Digits Image', small)

#cv2.waitKey(0)
#cv2.destroyAllWindows()


import cv2
import numpy as np
file = 'D:/bikram/machine learning/PracticeSets/sudoku/digits.png'
image = cv2.imread(file)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
small = cv2.pyrDown(image)
#cv2.imshow('Digits Image', small)

#cv2.waitKey(0)
#cv2.destroyAllWindows()
# Split the image to 5000 cells, each 20x20 size
# This gives us a 4-dim array: 50 x 100 x 20 x 20
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

# Convert the List data type to Numpy Array of shape (50,100,20,20)
x = np.array(cells)
print ("The shape of our cells array: " + str(x.shape))

# Split the full data set into two segments
# One will be used fro Training the model, the other as a test data set
train = x[:,:70].reshape(-1,400).astype(np.float32) # Size = (3500,400)
test = x[:,70:100].reshape(-1,400).astype(np.float32) # Size = (1500,400)
# Create labels for train and test data
k = [0,1,2,3,4,5,6,7,8,9]
train_labels = np.repeat(k,350)[:,np.newaxis]
test_labels = np.repeat(k,150)[:,np.newaxis]
print(train.shape,train_labels.shape)



# Initiate kNN, train the data, then test it with test data for k=3
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
#knn = cv2.ml.KNearest_create()
#knn.train(train, train_labels)
#ret, result, neighbors, distance = knn.find_nearest(test, k=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train, train_labels)
# Now we check the accuracy of classification
# For that, compare the result with test_labels and check which are wrong
model_score = knn.score(test, test_labels)

print("Accuracy is = %.2f" % model_score + "%")



def predict(df):
    predict = knn.predict(df.reshape(1,-1))[0]
    predict_proba = knn.predict_proba(df.reshape(1,-1))
    #print("probs",predict_proba)
    return predict, predict_proba[0][predict]
fileTemp = 'D:/bikram/machine learning/PracticeSets/sudoku/sudoku_01_01.png'

image = cv2.imread(fileTemp)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
redImg =255- cv2.resize(gray,(20,20))
cv2.imshow("20*20 Image",redImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
x,y = predict(redImg)
print(x,y)




np.count_nonzero(redImg)
fileTemp = 'D:/bikram/machine learning/PracticeSets/sudoku/BinarySudoku_'
for i in range(1,10,1):
    for j in range(1,10,1):
        name = '0'+str(i)+'_0'+str(j)+'.png'
        file = fileTemp + name
        image = cv2.imread(file)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        redImg =255- cv2.resize(gray[2:-3,2:-3],(20,20))
        #print(file,np.count_nonzero0(redImg))
        if(np.count_nonzero(redImg)>20):
            #cv2.imshow(name,redImg)
            x,y = predict(redImg)
            print(x,end=' ')
        else:
            print('_',end=' ')
    print('\n')
fileTemp = 'D:/bikram/machine learning/PracticeSets/sudoku/BinarySudoku.png'
image = cv2.imread(fileTemp)   
cv2.imshow("20*20 Image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()




import image_slicer
file = 'D:/bikram/machine learning/PracticeSets/sudoku/'
name = 'sudoku3.png'
image = cv2.imread(file+name)
ret,thresh1 = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
name = 'BinarySudoku3.png'
cv2.imwrite(file+name,thresh1)
#image_slicer.slice(file+name, 9*9)

fileTemp = 'D:/bikram/machine learning/PracticeSets/sudoku/BinarySudoku3_'
inputArr = [[0]*9 for i in range(0,9)]
for i in range(1,10,1):
    for j in range(1,10,1):
        name = '0'+str(i)+'_0'+str(j)+'.png'
        file = fileTemp + name
        image = cv2.imread(file)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        redImg =255- cv2.resize(gray[2:-3,2:-3],(20,20))
        #print(file,np.count_nonzero0(redImg))
        if(np.count_nonzero(redImg)>20):
            #cv2.imshow(name,redImg)
            x,y = predict(redImg)
            print(x,end=' ')
            inputArr[i-1][j-1] = x
        else:
            print('_',end=' ')
            
    print('\n')
    
    
    
    
    
    
def print_grid(arr): 
    for i in range(9): 
        for j in range(9): 
            print (arr[i][j],end='') 
        print ('\n') 
  
          
# Function to Find the entry in the Grid that is still  not used 
# Searches the grid to find an entry that is still unassigned. If 
# found, the reference parameters row, col will be set the location 
# that is unassigned, and true is returned. If no unassigned entries 
# remain, false is returned. 
# 'l' is a list  variable that has been passed from the solve_sudoku function 
# to keep track of incrementation of Rows and Columns 
def find_empty_location(arr,l): 
    for row in range(9): 
        for col in range(9): 
            if(arr[row][col]==0): 
                l[0]=row 
                l[1]=col 
                return True
    return False
  
# Returns a boolean which indicates whether any assigned entry 
# in the specified row matches the given number. 
def used_in_row(arr,row,num): 
    for i in range(9): 
        if(arr[row][i] == num): 
            return True
    return False
  
# Returns a boolean which indicates whether any assigned entry 
# in the specified column matches the given number. 
def used_in_col(arr,col,num): 
    for i in range(9): 
        if(arr[i][col] == num): 
            return True
    return False
  
# Returns a boolean which indicates whether any assigned entry 
# within the specified 3x3 box matches the given number 
def used_in_box(arr,row,col,num): 
    for i in range(3): 
        for j in range(3): 
            if(arr[i+row][j+col] == num): 
                return True
    return False
  
# Checks whether it will be legal to assign num to the given row,col 
#  Returns a boolean which indicates whether it will be legal to assign 
#  num to the given row,col location. 
def check_location_is_safe(arr,row,col,num): 
      
    # Check if 'num' is not already placed in current row, 
    # current column and current 3x3 box 
    return not used_in_row(arr,row,num) and not used_in_col(arr,col,num) and not used_in_box(arr,row - row%3,col - col%3,num) 
  
# Takes a partially filled-in grid and attempts to assign values to 
# all unassigned locations in such a way to meet the requirements 
# for Sudoku solution (non-duplication across rows, columns, and boxes) 
def solve_sudoku(arr): 
      
    # 'l' is a list variable that keeps the record of row and col in find_empty_location Function     
    l=[0,0] 
      
    # If there is no unassigned location, we are done     
    if(not find_empty_location(arr,l)): 
        return True
      
    # Assigning list values to row and col that we got from the above Function  
    row=l[0] 
    col=l[1] 
      
    # consider digits 1 to 9 
    for num in range(1,10): 
          
        # if looks promising 
        if(check_location_is_safe(arr,row,col,num)): 
              
            # make tentative assignment 
            arr[row][col]=num 
  
            # return, if sucess, ya! 
            if(solve_sudoku(arr)): 
                return True
  
            # failure, unmake & try again 
            arr[row][col] = 0
              
    # this triggers backtracking         
    return False 

grid=[[3,0,6,5,0,8,4,0,0], 
          [5,2,0,0,0,0,0,0,0], 
          [0,8,7,0,0,0,0,3,1], 
          [0,0,3,0,1,0,0,8,0], 
          [9,0,0,8,6,3,0,0,5], 
          [0,5,0,0,9,0,6,0,0], 
          [1,3,0,0,0,0,2,5,0], 
          [0,0,0,0,0,0,0,7,4], 
          [0,0,5,2,0,6,3,0,0]] 

if(solve_sudoku(inputArr)): 
    print_grid(inputArr) 
else:
    print ("No solution exists")
        