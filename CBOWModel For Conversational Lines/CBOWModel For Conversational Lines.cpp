// CBOWModel For Conversational Lines.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <assert.h>
#include "Net.h"
using namespace std;
string stopWords[119] = { "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now" };
int main()
{
	///////////Collect Stop words/////////////////////
	string stopWordList = "";
	for (int n = 0; n < 119; n++) {
		cout << "Processing Stop Words: " << n << "\r";
		//stopWordList += " " + stopWords[n] + " ";
	}
	cout << endl;
	vector<string> token;
	vector<vector<unsigned>> dialogueToken;
	//replace with file location of dialogue corpus
	string fileName = "C:\\Users\\Saurav\\Downloads\\Datasets\\cornell_movie_dialogs_corpus\\cornell movie-dialogs corpus\\movie_conversations.txt";
	ifstream fin_conv(fileName);
	cout << fin_conv.is_open() << endl;
	for (int l = 0; l < 100; l++) {
		cout << "Gathering Conversation Pairs: " << l << "\r";
		string line = "";
		getline(fin_conv, line);
		int countPlus = 0;
		while (countPlus < 18) {
			if (line[0] == '+') {
				countPlus++;
			}
			line.erase(0, 1);
		}
		string A = "";
		string B = "";
		line.erase(0, 3);
		int pos = 0;
		while (line[pos] != '\'') {
			A += line[pos];
			pos++;
		}
		pos+=4;
		while (line[pos] != '\'') {
			B += line[pos];
			pos++;
		}
		bool foundA = false;
		bool foundB = false;
		//replace with location of dialogue corpus
		string lines_file = "C:\\Users\\Saurav\\Downloads\\Datasets\\cornell_movie_dialogs_corpus\\cornell movie-dialogs corpus\\movie_lines.txt";
		ifstream fin_lines(lines_file);
		while (getline(fin_lines, line) && (!foundA || !foundB)) {
			int p = 0;
			string line_id = "";
			while (line[p] != ' ') {
				line_id += line[p];
				p++;
			}
			if (line_id == A || line_id == B) {
				if (line_id == A) {
					foundA = true;
				}
				else {
					foundB = true;
				}
				int countPlus2 = 0;
				while (countPlus2 < 24) {
					if (line[0] == '+') {
						countPlus2++;
					}
					line.erase(0, 1);
				}
				string word = "";
				dialogueToken.push_back({});
				for (int let = 0; let <= line.size(); let++) {
					char letter = tolower(line[let]);
					if (word.size() > 0 && (letter == ' ' || let == line.size())) {
						bool newWord = true;
						for (int t = 0; t < token.size(); t++) {
							if (token[t] == word) {
								newWord = false;
								dialogueToken.back().push_back(t);
							}
						}
						if (newWord && stopWordList.find(" " + word + " ") == string::npos) {
							token.push_back(word);
							dialogueToken.back().push_back(token.size() - 1);
						}
						word = "";
					}
					else if ((int)letter > 96 && (int)letter < 123) {
						word += letter;
					}
				}
			}
		}
	}
	cout << endl << dialogueToken.size() << endl;
	for (int d = 0; d < dialogueToken.size(); d+=2) {
		if (dialogueToken[d].empty() || dialogueToken[d + 1].empty()) {
			dialogueToken.erase(dialogueToken.begin() + d);
			dialogueToken.erase(dialogueToken.begin() + d);
			d -= 2;
			continue;
		}
	}
	vector<vector<double>> contextInputVectors;
	vector<unsigned> targetVals;
	int windowSize = 2;
	for (int l = 0; l < dialogueToken.size(); l++) {
		for (int w = 0; w < dialogueToken[l].size(); w++) {
			contextInputVectors.push_back({});
			int pos = windowSize;
			while (pos >= -windowSize) {
				if (pos == 0) {
					targetVals.push_back(dialogueToken[l][w]);
				}
				else if (pos + w < dialogueToken[l].size() && pos + w >= 0) {
					vector<double> vec(token.size(), 0.0);
					vec[dialogueToken[l][w + pos]] = 1.0;
					contextInputVectors.back().insert(contextInputVectors.back().end(), vec.begin(), vec.end());
				}
				else {
					vector<double> vec(token.size(), 0.0);
					contextInputVectors.back().insert(contextInputVectors.back().end(), vec.begin(), vec.end());
				}
				pos--;
			}
		}
	}
	assert(contextInputVectors.size() == targetVals.size());
	Net net(token.size() * 4, 1, 4);
	//Net net("CBOWModel10.txt");
	for (int e = 0; e < 100; e++) {
		for (int t = 0; t < targetVals.size(); t++) {
			net.feedForward(contextInputVectors[t]);
			vector<double> targ(token.size(), 0.0);
			targ[targetVals[t]] = 1.0;
			net.backProp(targ);
			cout << "Training:\t" << t + 1 << "\t/\t" << targetVals.size() << "\r";
		}
		cout << endl;
		int correctCount = 0;
		for (int t = targetVals.size() - 100; t < targetVals.size(); t++) {
			net.feedForward(contextInputVectors[t]);
			vector<double> resultVals;
			net.getResults(resultVals);
			double max = resultVals[0];
			unsigned pos = 0;
			for (int r = 0; r < resultVals.size(); r++) {
				if (max < resultVals[r]) {
					max = resultVals[r];
					pos = r;
				}
			}
			if (pos == targetVals[t]) {
				correctCount++;
			}
			cout << "Testing:\t" << correctCount << "\t/\t" << 100 << "\r";
		}
		cout << endl;
		net.storeNet("CBOWModel10.txt");
	}
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
