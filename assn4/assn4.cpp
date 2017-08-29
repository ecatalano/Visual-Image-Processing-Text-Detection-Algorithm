#include <iostream>
using namespace std;

#include <stdio.h>
#include <math.h>
#include <fstream>
#include <algorithm>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"

using namespace cv;
using namespace cv::ml;

Mat convertToGreyscale(Mat image, string name){
	//The equation used in the paper was:
	//Y = 0.2126R+0.7152G+0.0722B
	
	Mat retImage = imread(name);
	
	for(int i = 0; i < image.rows; i++){
		for(int j = 0; j < image.cols; j++){
			int sum = 0;
			sum += image.at<Vec3b>(i,j)[0] * .0722;
			sum += image.at<Vec3b>(i,j)[1] * .7152;
			sum += image.at<Vec3b>(i,j)[2] * .2126;
				
			retImage.at<Vec3b>(i,j)[0] = sum;
			retImage.at<Vec3b>(i,j)[1] = sum;
			retImage.at<Vec3b>(i,j)[2] = sum;
		}
	}

	return retImage;
}

Mat binarizeImage(Mat image, string name){
	Mat retImage = imread(name);
	int max1 = 0;
	int max2 = 0;

	int hist[255];	

	for(int i = 0; i < 255; i++){
		hist[i] = 0;
	}

	for(int i = 0; i < image.rows; i++){
		for(int j = 0; j < image.cols; j++){
			int pixel = image.at<Vec3b>(i,j)[0];
			hist[pixel]++;
		}
	}
	//find the peak (background)
	for(int i = 0; i < 255; i++){
		int value = hist[i];
		if(value > hist[max1]) max1 = i;
	}
	
	for(int i = 0; i < image.rows; i++){
		for(int j = 0; j < image.cols; j++){
			int pixel = image.at<Vec3b>(i,j)[0];
			if(pixel < max1 + 45 && pixel > max1 - 65){
				retImage.at<Vec3b>(i,j)[0] = 255;
				retImage.at<Vec3b>(i,j)[1] = 255;
				retImage.at<Vec3b>(i,j)[2] = 255;
			}
			else{
				retImage.at<Vec3b>(i,j)[0] = 0;
				retImage.at<Vec3b>(i,j)[1] = 0;
				retImage.at<Vec3b>(i,j)[2] = 0;
			}


		}
	}
	
	return retImage;
	
}
Mat removeNoise(Mat image, string name){
	for(int i = 1; i < image.rows-1; i++){
		for(int j = 1; j < image.cols-1; j++){
			int pixel = image.at<Vec3b>(i,j)[0];
			int pixelNorth = image.at<Vec3b>(i-1,j)[0];
			int pixelSouth = image.at<Vec3b>(i+1,j)[0];
			int pixelWest = image.at<Vec3b>(i,j-1)[0];
			int pixelEast = image.at<Vec3b>(i,j+1)[0];
			if(pixel == 0 && pixelNorth !=0 && pixelSouth!=0 && pixelEast!=0 && pixelWest != 0 ){
				image.at<Vec3b>(i,j)[0] = 255;
				image.at<Vec3b>(i,j)[1] = 255;
				image.at<Vec3b>(i,j)[2] = 255;

			}
		}
	}
	return image;
}

string recognizeCharacter(int array[15][15], bool training){
	
	float maxDistance = 0;
	float distance = 0;
	float sizeOfTrack;
	int trackNumber;
	string line;

	string currentLetter;
	string matchedLetter;
	
	ofstream trainingFile;
	trainingFile.open("trainingFile.txt", std::ofstream::app);
	ifstream trainedFile("trainingFile.txt");

	int sectorMatrix[15][15];
	float distanceMatrix[15][15];
	int sectorTrackMatrix[8][5];
	int trainedDataMatrix[8][5];

	for(int i = 0; i < 8; i++){
		for(int j = 0; j < 5; j++){
			sectorTrackMatrix[i][j] = 0;
			trainedDataMatrix[i][j] = 0;
		}
	}
		
	for(int i = 0; i < 15; i++){
		for(int j = 0; j < 15; j++){
			sectorMatrix[i][j] = 0;
			distanceMatrix[i][j] = 0;
		}
	}
	if(training == true){
		for(int i = 0; i < 15; i++){
			for(int j = 0; j < 15; j++){
				printf("%d ", array[i][j]);
			}
			printf("\n");
		}
		printf("-----------------------------\n");
	}

	//1. Identify center of matrix
	//15x15 matrix... center is at [8][8];

	//2. Calculate radius (rad) by finding pixel with maximum distance from center
	//Formula: Dist = sqrt((y2-y1)*(y2-y1) + (x2-x1)*(x2-x1))
	for(int i = 0; i < 15; i++){
		for(int j = 0; j < 15; j++){
			int x1;
			int x2;
			int y1;
			int y2;

			int pixel = array[i][j];
			if(j <= 8){
				x2 = 8;
				x1 = j;
			}
			else{
				x2 = j;
				x1 = 8;
			}
			if( i <= 8){
				y2 = 8;
				y1 = i;
			}
			else{
				y2 = i;
				y1 = 8;
			}
			
			if(pixel == 0){
				distance = sqrt((y2-y1)*(y2-y1) + (x2-x1)*(x2-x1));
				distanceMatrix[i][j] = distance;
				if(distance > maxDistance) maxDistance = distance;
			}
		}
	}
	//printf("Max Distance: %f\n", maxDistance);
	
	//3. Perform (rad % 5) to identify size of each imaginary track
	sizeOfTrack = (float)maxDistance / 5;
	//printf("Size of Track: %.1f\n", sizeOfTrack);
	
	//4. Identify Imaginary Sectors
	//Can use x = 8, y = 8, and x = y and [0][8], [1][7]... [8][0] line to divide sectors
	for(int i = 0; i < 15; i++){
		for(int j = 0; j < 15; j++){
			sectorMatrix[i][j] = 0;
			if(i < 7){
			//top half
				if(j < 7){
				//left half
					if(j > i){
					//top
						sectorMatrix[i][j] = 8;
					}
					else if(j < i){
					//bot
						sectorMatrix[i][j] = 7;
					}

				}
				else if(j > 7){
				//right half
					if(j+i < 14){
					//top

						sectorMatrix[i][j] = 1;
					}
					else if(j+i > 14){
					//bot
						sectorMatrix[i][j] = 2;
					}
				}
			}
			else if(i > 7){
			//bottom half
				if(j < 7){
				//left half
					if(j+i < 14){
					//top
						sectorMatrix[i][j] = 6;
					}
					else if(j+i > 14){
					//bot
						sectorMatrix[i][j] = 5;
					}
				}
				else if(j > 7){
				//right half
					if(j > i){
					//top
						sectorMatrix[i][j] = 3;
					}
					else if(j < i){
					//bot
						sectorMatrix[i][j] = 4;
					}

				}
			}
			if(j == 7 && i < 7){
				sectorMatrix[i][j] = 8;
			}
			else if(j == 7 && i > 7){
				sectorMatrix[i][j] = 4;
			}
			else if(i == 7 && j < 7){
				sectorMatrix[i][j] = 6;
			}
			else if(i == 7 && j > 7){
				sectorMatrix[i][j] = 2;
			}
			else if(i == j && i < 7){
				sectorMatrix[i][j] = 7;
			}
			else if(i == j && i > 7){
				sectorMatrix[i][j] = 3;
			}
			else if(i + j == 14 && i < 7){
				sectorMatrix[i][j] = 1;
			}
			else if(i + j == 14 && i > 7){
				sectorMatrix[i][j] = 5;
			}

			/*for(int i = 0; i < 15; i++){
				for(int j = 0; j < 15; j++){
					printf("%d ", sectorMatrix[i][j]);
				}
				printf("\n");
			}
			printf("-------------------------------\n");*/
		}
	}

	//5. Generate track-sector matrix by calculating number of 1's 
	//in each intersection of sector and track
	for(int i = 0; i < 15; i++){
		for(int j = 0; j < 15; j++){
			float tempDistance = distanceMatrix[i][j];
			int tempSector = sectorMatrix[i][j];
			int tempTrack = 0;
			if(tempDistance <= sizeOfTrack){
				tempTrack = 1;
			}
			else if(tempDistance <= sizeOfTrack * 2){
				tempTrack = 2;
			}
			else if(tempDistance <= sizeOfTrack * 3){
				tempTrack = 3;
			}
			else if(tempDistance <= sizeOfTrack * 4){
				tempTrack = 4;
			}
			else if(tempDistance <= sizeOfTrack * 5){
				tempTrack = 5;
			}

			sectorTrackMatrix[tempSector-1][tempTrack-1]++;
		}
	}

	if(training == false){
		while(getline(trainedFile, line)){
			bool isMatching = true;

			std::istringstream iss(line);
			iss >> currentLetter >> trainedDataMatrix[0][0] >> trainedDataMatrix[0][1] >> trainedDataMatrix[0][2] >> trainedDataMatrix[0][3] >> trainedDataMatrix[0][4] >> trainedDataMatrix[1][0] >> trainedDataMatrix[1][1] >> trainedDataMatrix[1][2] >> trainedDataMatrix[1][3] >> trainedDataMatrix[1][4] >> trainedDataMatrix[2][0] >> trainedDataMatrix[2][1] >> trainedDataMatrix[2][2] >> trainedDataMatrix[2][3] >> trainedDataMatrix[2][4] >> trainedDataMatrix[3][0] >> trainedDataMatrix[3][1] >> trainedDataMatrix[3][2] >> trainedDataMatrix[3][3] >> trainedDataMatrix[3][4] >> trainedDataMatrix[4][0] >> trainedDataMatrix[4][1] >> trainedDataMatrix[4][2] >> trainedDataMatrix[4][3] >> trainedDataMatrix[4][4] >> trainedDataMatrix[5][0] >> trainedDataMatrix[5][1] >> trainedDataMatrix[5][2] >> trainedDataMatrix[5][3] >> trainedDataMatrix[5][4] >> trainedDataMatrix[6][0] >> trainedDataMatrix[6][1] >> trainedDataMatrix[6][2] >> trainedDataMatrix[6][3] >> trainedDataMatrix[6][4] >> trainedDataMatrix[7][0] >> trainedDataMatrix[7][1] >> trainedDataMatrix[7][2] >> trainedDataMatrix[7][3] >> trainedDataMatrix[7][4];

			for(int i = 0; i < 8; i++){
				for(int j = 0; j < 5; j++){
					if(sectorTrackMatrix[i][j]!=trainedDataMatrix[i][j]){
						isMatching = false;
					}
				}
			}
			if(isMatching == true){
				//printf("Matched Letter: %s\n", currentLetter.c_str());
				matchedLetter = currentLetter;
				return matchedLetter;
			}
		}
	}

	else{
		printf("What Letter is Being Displayed?\n");
		cin >> currentLetter;
		trainingFile << currentLetter << " ";
		
		for(int i = 0; i < 8; i++){
			for(int j = 0; j < 5; j++){
				trainingFile << sectorTrackMatrix[i][j] << " ";
			}
		}
		trainingFile << "\n";
		trainingFile.close();
	}

	return matchedLetter;
}
float findAverageWidth(Mat image){
	vector<int> tops;
	vector<int> bots;
	vector<int> lefts;
	vector<int> rights;
	vector<int> spaces;

	bool foundTop = false;
	bool foundLeft = false;
	bool foundLetter = false;

	bool allWhite = true;
	int lineCount = 0;
	int letterCount = 0;

	//for counting number of vertical white lines (detecting spaces)
	int whiteLineCount = 0;
	int spaceCount = 0;

	int whiteLineNum = 0;

	int top = 0;
	int bot = 0;
	int left = 0;
	int right = 0;
	int nextSpace = 0;

	float averageWidth = 0;

	for(int i = 0; i < image.rows; i++){
		for(int j = 0; j < image.cols; j++){
			//want to find top of rows..
			int pixel = image.at<Vec3b>(i,j)[0];

			if(foundTop == false){	
				if(pixel == 0){
					top = i;
					foundTop = true;
				}
			}
			else{
				//if any pixel within the row is not white
				//make allWhite == false
				if(pixel != 255){
					allWhite = false;
				}
			}
		}
		//end of loop, 
		if(allWhite == true && foundTop == true) {
			bot = i;
			foundTop = false;
			tops.push_back(top);
			bots.push_back(bot);
			lineCount++;			
		}
		allWhite = true;
	}
	if(lineCount == 0) printf("Could Not Find Any Lines...\n");
	for(int c = 0; c < lineCount; c++){
		//printf("Top: %d Bottom: %d\n", tops.at(c), bots.at(c));
		top = tops.at(c);
		bot = bots.at(c);
		letterCount = 0;
		lefts.clear();
		rights.clear();
		spaces.clear();
		for(int j = 0; j < image.cols; j++){
			for(int i = top; i < bot; i++){
				int pixel = image.at<Vec3b>(i,j)[0];
				if(foundLeft == false){
					if(pixel == 0){
						left = j;
						foundLeft = true;
					}
				}
				else{
					if(pixel !=255){
						allWhite = false;
					}
				}
			}
			if(foundLeft == true && allWhite == true){
				right = j;
				lefts.push_back(left);
				rights.push_back(right);
				foundLeft = false;
				letterCount++;
				spaceCount++;
			}
			else if(foundLeft == false && allWhite == true){
				whiteLineCount++;
			}
			allWhite = true;	
		}
	}
	

	for(int i = 0; i < letterCount; i++){
		averageWidth+=rights.at(i) - lefts.at(i);
	}
	averageWidth/= letterCount;
	//printf("Average Width of Letter: %f\n", averageWidth);

	return averageWidth;

}

string extractFeatures(Mat image){

	vector<int> tops;
	vector<int> bots;
	vector<int> lefts;
	vector<int> rights;
	vector<int> spaces;

	bool foundTop = false;
	bool foundLeft = false;
	bool foundLetter = false;

	bool allWhite = true;
	int lineCount = 0;
	int letterCount = 0;

	//for counting number of vertical white lines (detecting spaces)
	int whiteLineCount = 0;

	int whiteLineNum = 0;

	int top = 0;
	int bot = 0;
	int left = 0;
	int right = 0;
	int nextSpace = 0;

	float averageWidth = findAverageWidth(image);
	
	string input;
	string recognizedString = "";
	bool training = false;

	printf("Training? (y/n)\n");
	cin >> input;

	if(input == "y"){
		training = true;
	}

	for(int i = 0; i < image.rows; i++){
		for(int j = 0; j < image.cols; j++){
			//want to find top of rows..
			int pixel = image.at<Vec3b>(i,j)[0];

			if(foundTop == false){	
				if(pixel == 0){
					top = i;
					foundTop = true;
				}
			}
			else{
				//if any pixel within the row is not white
				//make allWhite == false
				if(pixel != 255){
					allWhite = false;
				}
			}
		}
		//end of loop, 
		if(allWhite == true && foundTop == true) {
			bot = i;
			foundTop = false;
			tops.push_back(top);
			bots.push_back(bot);
			lineCount++;			
		}
		allWhite = true;
	}
	if(lineCount == 0) printf("Could Not Find Any Lines...\n");
	for(int c = 0; c < lineCount; c++){
		//printf("Top: %d Bottom: %d\n", tops.at(c), bots.at(c));
		top = tops.at(c);
		bot = bots.at(c);
		letterCount = 0;
		lefts.clear();
		rights.clear();
		spaces.clear();
		for(int j = 0; j < image.cols; j++){
			for(int i = top; i < bot; i++){
				int pixel = image.at<Vec3b>(i,j)[0];
				if(foundLeft == false){
					if(pixel == 0){
						left = j;
						foundLeft = true;
					}
				}
				else{
					if(pixel !=255){
						allWhite = false;
					}
				}
			}
			if(foundLeft == true && allWhite == true){
				right = j;
				lefts.push_back(left);
				rights.push_back(right);
				foundLeft = false;
				letterCount++;
				whiteLineCount = 0;
			}
			else if(foundLeft == false && allWhite == true){
				whiteLineCount++;
				//printf("WhiteLineCount: %d\n", whiteLineCount);

				if(whiteLineCount == (int)(averageWidth * .8)){
					spaces.push_back(j);
					whiteLineCount = 0;
				}
			}
			allWhite = true;	
		}
		//all lefts and rights of letters should be found..
		if(letterCount == 0) printf("Could Not Find Any Letters.\n");
		for(int x = 0; x < letterCount; x++){
			left = lefts.at(x);
			right = rights.at(x);

			allWhite = true;
			whiteLineNum = 0;

			int newBot = 0;
			int newTop = 0;
			bool foundNewTop = false;
			bool foundNewBot = false;
			//printf("Top: %d Bottom: %d Left: %d Right: %d\n", top, bot, left, right);
			if(right-left > 0){
				
				if(spaces.size() > 0){
					nextSpace = spaces.at(0);
					//printf("Next Space: %d\n", nextSpace);
					//printf("Left: %d Right: %d Space: %d\n", left, right, nextSpace);
					if(nextSpace < left){
						recognizedString.append(" ");
						spaces.erase(spaces.begin());	
					}
				} 

				Mat extractedLetter(image, Rect(left,top,right-left, bot-top));
				//Now we need to figure out if there is any white space
				//at bottom and top of extracted letter...
				//first need to find white space of top...
				for(int i = 0; i < bot-top; i++){
					for(int j = 0; j < right-left; j++){
						if(foundNewTop == false && extractedLetter.at<Vec3b>(i,j)[0]==0){
							foundNewTop = true;
							newTop = top + i;
							//printf("New Top: %d\n", newTop);
						}
					}
				}
				Mat topCrop(image, Rect(left, newTop, right-left, bot-newTop));
				extractedLetter = topCrop;

				for(int i = bot-newTop; i > 0; i--){
					for(int j = 0; j < right-left; j++){
						if(topCrop.at<Vec3b>(i,j)[0]==0 && foundNewBot == false){
							newBot = newTop + i;
							foundNewBot = true;
							//printf("newTop: %d bot: %d new bot: %d i: %d\n", newTop, bot, newBot, i);
						}
					}
				}
				
				

				Mat cropBot(image, Rect(left,newTop,right-left,newBot-newTop));
				extractedLetter = cropBot;
				
				imshow("Extracted Letter", extractedLetter);
				waitKey(1);
	
				Mat resizedLetter;
				Size size(15,15);
				resize(extractedLetter, resizedLetter, size);
					
				//imshow("Resized Letter", resizedLetter);
				//waitKey(0);

				//now that we have the resized 15x15 letter
				//we have to binarize it into an array of 1's and 0's...
				int binaryArray[15][15];
				for(int i = 0; i < 15; i++){
					for(int j = 0; j < 15; j++){
						int pixel = resizedLetter.at<Vec3b>(i,j)[0];
						if(pixel == 0) binaryArray[i][j] = 0;
						else binaryArray[i][j] = 1;
					}
				}
				
				string recognizedChar = recognizeCharacter(binaryArray, training);
				recognizedString.append(recognizedChar);
			}
		}
		
	}
	return recognizedString;
}

string removeSpaces(string input){
	input.erase(std::remove(input.begin(),input.end(),' '),input.end());
	return input;
}

string compareIngredients(string input){

	string retVal = "";

	string animalIngredients[] = {"Albumin", "Alcloxa", "Aldioxa", "Aliphatic Alcohol", "Allantoin", "Alligator Skin", "Alpha-Hydroxy Acids", "Ambergris", "Amerchol L101", "Amino Acids", "Aminosuccinate Acid", "Angora", "Animal Fats and Oils", "Animal Hair", "Arachidonic Adic", "Arachidyl Propionate", "Bee Pollen", "Bee Products", "Beeswax", "Honeycomb", "Biotin", "Vitamin H", "Vitamin B Factor", "Blood", "Boar Bristles", "Bone Char", "Bone Meal", "Calciferol", "Calfskin", "Caprylamine Oxide", "Caprylic Triglyceride", "Carbamide", "Carmine", "Cochineal", "Carminic Acid", "Carotene", "Provitamin A", "Beta Carotene", "Casein", "Caseinate", "Sodium Caseinate", "Caseinate", "Cashmere", "Castor", "Castoreum", "Catgut", "Cera Flava", "Cerebrosides", "Cetyl Alcohol", "Cetyl Palmitate", "Chitosan", "Cholesterin", "Cholesterol", "Choline Bitartrate", "Civet", "Cochineal", "Cod Liver Oil", "Collagen", "Colors", "Dyes", "Corticosteroid", "Cysteine, L-Form", "Cystine", "Dexpanthenol", "Diglycerides", "Dimethyl Stearamine", "Down", "Duodenum Substances", "Egg Protein", "Elastin", "Emu Oil", "Ergocalciferol", "Ergosterol", "Estradoil", "Estrogen", "Fatty Acids", "FD&C Colors", "Feathers", "Fish Liver Oil", "Fish Oil", "Fish Scales", "Fur", "Gel", "Gelatin", "Glucose Tyrosinase", "Glycerides", "Glycerin", "Glycerol", "Glyceryls", "Glycreth-26", "Guanine", "Pearl Essence", "Hide Glue", "Honey", "Honeycomb", "Horsehair", "Hyaluronic Acid", "Hydrocortisone", "Hydrolyzed Animal Protein", "Imidazolidinyl Urea", "Insulin", "Isinglass", "Isopropyl Lanolate", "Isopropyl Myristate", "Isopropyl Palmitate", "Keratin", "Lactic Acid", "Lactose", "Laneth", "Lanogene", "Lanolin", "Lanolin Acids", "Wool Food", "Wool Wax", "Lanolin Alcohol", "Lanosterols", "Lard", "Leather", "Suede", "Calfskin", "Sheepskin", "Skin", "Lecithin", "Choline Bitartrate", "Linoleic Acid", "Lipase", "Lipids", "Lipoids", "Marine Oil", "Methionine", "Milk Protein", "Milk","Milk Protein Concentrate", "Mink Oil", "Monoglycerides", "Glycerides", "Musk (Oil)", "Myristal Ether Sulfate", "Myristic Acid", "Myristyls", "Natural Sources", "Nucleic Acids", "Ocenol", "Octyl Dodecanol", "Oleic Acid", "Oils", "Oleths", "Oleyl Alcohol", "Ocenol", "Oley; Arachidate", "Oleyl Imidazoline", "Oleyl Myristate", "Oleyl Oleate", "Oleyl Stearate", "Palmitamide", "Palmitamine", "Palmitate", "Palmitic Acid", "Panthenol", "Dexpanthenol", "Vitamin B-Complex Factor", "Provitamin B-5", "Panthenyl", "Pepsin", "Placenta", "Placenta Polypeptides Protein", "Afterbirth", "Polyglycerol", "Polypeptides", "Polysorbates", "Pristane", "Progesterone", "Propolis", "Provitamin A", "Provitamin B-5", "Provitamin D-2", "Rennet", "Rennin", "Resinous Glaze", "Retinol", "Ribonucleic Acid", "RNA", "Ribonucleic Acid", "Royal Jelly", "Sable Brushes", "Sea Turtle Oil", "Shark Liver Oil", "Sheepskin", "Shellac", "Resinous Glaze", "Silk", "Silk Powder", "Snails", "Sodium Caseinate", "Sodium Steroyl Lactylate", "Sodium Tallowate", "Spermaceti", "Cetyl Parlmate", "Sperm Oil", "Sponge", "Squalane", "Squalene", "Stearamide", "Stearamine", "Stearamine Oxide", "Stearates", "Stearic Acid", "Stearic Hydrazide", "Stearone", "Stearoxytrimethylsilane", "Stearoxyl Lactylic Acid", "Stearyl Acetate", "Stearyl Alcohol", "Sterols", "Stearyl Betaine", "Stearyl Caprylate", "Stearyl Citrate", "Stearyldimethyl Amine", "Stearyl Glycyrrhetinate", "Stearyl Heptanoate", "Stearyl Imidazoline", "Stearyl Octanoate", "Stearyl Stearate", "Steroids", "Sterols", "Suede", "Tallow", "Tallow Fatty Alcohol", "Stearic Acid", "Tallow Acid", "Tallow Amide", "Tallow Amine", "Talloweth-6", "Tallow Glycerides", "Tallow Imidazoline", "Triterpene Alcohols", "Uric Acid", "Vitamin A", "Vitamin B-Complex Factor", "Vitamin B Factor", "Vitamin B12", "Vitamin D", "Ergocalciferol", "Vitamin D2", "Ergosterol", "Provitamin D2", "Calciferol", "Vitamin D3", "Vitamin H", "Wax", "Whey", "Wool", "Wool Fat", "Wool Wax"};
	string replacedIngredients[254];
	
	size_t size = sizeof(animalIngredients) / sizeof(animalIngredients[0]);

	for(int i = 0; i < size; i++){
		string tempString = animalIngredients[i];
		size_t position = 0;

		//what to replace with
		string replacedString("l");

		//what to replace
		//i's will be replaced with l's
  		string stringToReplace("i");

  		//First time, we will see if we find the string
  		int pos = tempString.find(stringToReplace);

  		while(pos != string::npos){
    			tempString.erase(pos,stringToReplace.length());
    			tempString.insert(pos,replacedString);
   			pos = tempString.find(stringToReplace);
  		}
		replacedIngredients[i] = tempString;	
	}
	for(int i = 0; i < size; i++){
		string tempString = animalIngredients[i];
		transform(tempString.begin(), tempString.end(), tempString.begin(), ::tolower);

		string tempString2 = replacedIngredients[i];
		transform(tempString2.begin(), tempString2.end(), tempString2.begin(), ::tolower);
		
		transform(input.begin(), input.end(), input.begin(), ::tolower);
		
		if (removeSpaces(input).find(removeSpaces(tempString)) != std::string::npos) {
    			retVal = animalIngredients[i];
		}

		else if (removeSpaces(input).find(removeSpaces(tempString2)) != std::string::npos) {
    			retVal = replacedIngredients[i];
		}
	}
	
	return retVal;
}

int main(int argc, char** argv){
	
	string name1 = "test5.png";
	Mat image1 = imread(name1);
	//imshow("Original Image", image1);

	int count = 0;

	//The basic algorithm:
	//greyscaling
	Mat greyImage = convertToGreyscale(image1, name1);
	//imshow("Grey Image", greyImage);
	//also need to binarize the image and remove noise(not shown in paper) to extract features.
	Mat binaryImage = binarizeImage(greyImage, name1);
	Mat lessNoisyImage = removeNoise(binaryImage, name1);

	imshow("Binary Image", binaryImage);

	//feature extraction
	string extractedString = extractFeatures(lessNoisyImage);

	printf("Extracted String: %s\n", extractedString.c_str());
	
	vector<string> splitString;

	std::istringstream ss(extractedString);
	std::string token;

	while(std::getline(ss, token, ',')) {
   		std::cout << removeSpaces(token) << '\n';
   		splitString.push_back(token);

	}
	
	for(int i = 0; i < splitString.size(); i++){
		//printf("%s\n", splitString.at(i).c_str());
		string badIngredient = compareIngredients(splitString.at(i));
		if(badIngredient != ""){
			printf("Non-Vegan Ingredient: %s\n", badIngredient.c_str());
			count++;
		}
	}
	if(count==0){
		printf("Did not find any non-vegan ingredients.\n");
	}


	//waitKey();
	return 0;
}