#include <iostream>

#include "helpers.cpp"
#include <string>

using namespace GARCoder;
using namespace std;
int main(int numb_arg1, char * arg1[])
{
	string command;
	string device;
	string codecMode;
	string file1;
	string file2;

	while (true) 
	{
		cout << endl;
		cout << "Enter command "<< endl << "\t'c' to compress"<<endl<< "\t'd' to decompress"<<endl<<"\t'cd' to compress, decompress and compare"<<endl<<"\t'q' to quit" << endl;
		cin >> command;
		cout << "Choose codec mode" << endl << "\t's' for static binary" << endl << "\t'a' for adaptive binary" << endl << "\t'ca' for context adaptive binary (context is byte-length prefix)" << endl << "\t'ab' for adaptive with byte-length symbols" << endl;
		cin >> codecMode;
		cout << "Choose device " << endl << "\t'c' for CPU" << endl << "\t'g' for GPU" << endl << "\t'cg' for using both and comparing results" << endl;
		cin >> device;
		cout << "Enter file path ('1' - '4' for predefined test files)" << endl;
		cin >> file1;
		if (file1 == "1")
			file1 = "../../Tests/small.txt";
		if (file1 == "2")
			file1 = "../../Tests/test.txt";
		if (file1 == "3")
			file1 = "../../Tests/test.png";
		if (file1 == "4")
			file1 = "../../Tests/test.wav";

		cout << "Enter result file path ('d' to generate default filename automatically)" << endl;
		cin >> file2;
		CPUCodec cpuCodec;
		GPUCodec gpuCodec;
		StaticBinaryModel smodel;
		AdaptiveBinaryModel amodel;
		AdaptiveByteModel abmodel;
		ContextAdaptiveBinaryModel camodel;
		try
		{
			if (command == "c") {
				if (file2 == "d") {
					file2 = file1 + ".arc";
				}

				if (codecMode == "s") {
					float prob = Helpers::GetZerosProbability(file1.c_str());
					cout << "Zero probability: " << prob << std::endl;
					smodel.SetZeroProbability(prob);
					if (device == "c") {
						Helpers::EncodeFile(file1.c_str(), file2.c_str(), cpuCodec, smodel);
						continue;
					}
					else if (device == "g") {
						Helpers::EncodeFile(file1.c_str(), file2.c_str(), gpuCodec, smodel);
						continue;
					}
					else if (device == "cg") {
						string cpuFile = file2 + ".cpu";
						string gpuFile = file2 + ".gpu";
						Helpers::EncodeFile(file1.c_str(), cpuFile.c_str(), cpuCodec, smodel);
						Helpers::EncodeFile(file1.c_str(), gpuFile.c_str(), gpuCodec, smodel);
						cout << "Comparing files encoded by CPU and GPU..." << endl;
						Helpers::CompareFiles(cpuFile.c_str(), gpuFile.c_str());
						continue;
					}
				}
				if (codecMode == "a") {
					if (device == "c") {
						Helpers::EncodeFile(file1.c_str(), file2.c_str(), cpuCodec, amodel);
						continue;
					}
					else if (device == "g") {
						Helpers::EncodeFile(file1.c_str(), file2.c_str(), gpuCodec, amodel);
						continue;
					}
					else if (device == "cg") {
						string cpuFile = file2 + ".cpu";
						string gpuFile = file2 + ".gpu";
						Helpers::EncodeFile(file1.c_str(), cpuFile.c_str(), cpuCodec, amodel);
						Helpers::EncodeFile(file1.c_str(), gpuFile.c_str(), gpuCodec, amodel);
						cout << "Comparing files encoded by CPU and GPU..." << endl;
						Helpers::CompareFiles(cpuFile.c_str(), gpuFile.c_str());
						continue;
					}
				}
				if (codecMode == "ab") {
					if (device == "c") {
						Helpers::EncodeFile(file1.c_str(), file2.c_str(), cpuCodec, abmodel);
						continue;
					}
					else if (device == "g") {
						Helpers::EncodeFile(file1.c_str(), file2.c_str(), gpuCodec, abmodel);
						continue;
					}
					else if (device == "cg") {
						string cpuFile = file2 + ".cpu";
						string gpuFile = file2 + ".gpu";
						Helpers::EncodeFile(file1.c_str(), cpuFile.c_str(), cpuCodec, abmodel);
						Helpers::EncodeFile(file1.c_str(), gpuFile.c_str(), gpuCodec, abmodel);
						cout << "Comparing files encoded by CPU and GPU..." << endl;
						Helpers::CompareFiles(cpuFile.c_str(), gpuFile.c_str());
						continue;
					}
				}
				if (codecMode == "ca") {
					if (device == "c") {
						Helpers::EncodeFile(file1.c_str(), file2.c_str(), cpuCodec, camodel);
						continue;
					}
					else if (device == "g") {
						Helpers::EncodeFile(file1.c_str(), file2.c_str(), gpuCodec, camodel);
						continue;
					}
					else if (device == "cg") {
						string cpuFile = file2 + ".cpu";
						string gpuFile = file2 + ".gpu";
						Helpers::EncodeFile(file1.c_str(), cpuFile.c_str(), cpuCodec, camodel);
						Helpers::EncodeFile(file1.c_str(), gpuFile.c_str(), gpuCodec, camodel);
						cout << "Comparing files encoded by CPU and GPU..." << endl;
						Helpers::CompareFiles(cpuFile.c_str(), gpuFile.c_str());
						continue;
					}
				}
			}
			if (command == "cd") {
				string file3;
				cout << "Enter decompressed file name ('d' for default)" << endl;
				cin >> file3;
				if (file2 == "d") {
					file2 = file1 + ".arc";
				}
				if (file3 == "d") {
					file3 = file2 + ".decompressed";
				}
				
				if (codecMode == "s") {
					float prob = Helpers::GetZerosProbability(file1.c_str());
					cout << "Zero probability: " << prob << std::endl;
					smodel.SetZeroProbability(prob);
					if (device == "c") {
						Helpers::EncodeFile(file1.c_str(), file2.c_str(), cpuCodec, smodel);
						Helpers::DecodeFile(file2.c_str(), file3.c_str(), cpuCodec, smodel);
						cout << "Comparing source and decoded files..." << endl;
						Helpers::CompareFiles(file1.c_str(), file3.c_str());
						continue;
					}
					else if (device == "g") {
						Helpers::EncodeFile(file1.c_str(), file2.c_str(), gpuCodec, smodel);
						Helpers::DecodeFile(file2.c_str(), file3.c_str(), cpuCodec, smodel);
						Helpers::CompareFiles(file1.c_str(), file3.c_str());
						continue;
					}
					else if (device == "cg") {
						string cpuFile = file2 + ".cpu";
						string gpuFile = file2 + ".gpu";
						string cpuDecompressed = cpuFile + ".decompressed";
						string gpuDecompressed = gpuFile + ".decompressed";

						cout << "CPU" << endl;
						Helpers::EncodeFile(file1.c_str(), cpuFile.c_str(), cpuCodec, smodel);
						Helpers::DecodeFile(cpuFile.c_str(), cpuDecompressed.c_str(), cpuCodec, smodel);
						cout << "Comparing source and decoded files..." << endl;
						Helpers::CompareFiles(file1.c_str(), cpuDecompressed.c_str());
						
						cout << "GPU" << endl;
						Helpers::EncodeFile(file1.c_str(), gpuFile.c_str(), gpuCodec, smodel);
						Helpers::DecodeFile(gpuFile.c_str(), gpuDecompressed.c_str(), cpuCodec, smodel);
						cout << "Comparing source and decoded files..." << endl;
						Helpers::CompareFiles(file1.c_str(), gpuDecompressed.c_str());
						continue;
					}
				}
				if (codecMode == "a") {
					if (device == "c") {
						Helpers::EncodeFile(file1.c_str(), file2.c_str(), cpuCodec, amodel);
						Helpers::DecodeFile(file2.c_str(), file3.c_str(), cpuCodec, amodel);
						cout << "Comparing source and decoded files..." << endl;
						Helpers::CompareFiles(file1.c_str(), file3.c_str());
						continue;
					}
					else if (device == "g") {
						Helpers::EncodeFile(file1.c_str(), file2.c_str(), gpuCodec, amodel);
						Helpers::DecodeFile(file2.c_str(), file3.c_str(), cpuCodec, amodel);
						Helpers::CompareFiles(file1.c_str(), file3.c_str());
						continue;
					}
					else if (device == "cg") {
						string cpuFile = file2 + ".cpu";
						string gpuFile = file2 + ".gpu";
						string cpuDecompressed = cpuFile + ".decompressed";
						string gpuDecompressed = gpuFile + ".decompressed";

						cout << "CPU" << endl;
						Helpers::EncodeFile(file1.c_str(), cpuFile.c_str(), cpuCodec, amodel);
						Helpers::DecodeFile(cpuFile.c_str(), cpuDecompressed.c_str(), cpuCodec, amodel);
						cout << "Comparing source and decoded files..." << endl;
						Helpers::CompareFiles(file1.c_str(), cpuDecompressed.c_str());

						cout << "GPU" << endl;
						Helpers::EncodeFile(file1.c_str(), gpuFile.c_str(), gpuCodec, amodel);
						Helpers::DecodeFile(gpuFile.c_str(), gpuDecompressed.c_str(), cpuCodec, amodel);
						cout << "Comparing source and decoded files..." << endl;
						Helpers::CompareFiles(file1.c_str(), gpuDecompressed.c_str());
						continue;
					}
				}
				if (codecMode == "ab") {
					if (device == "c") {
						Helpers::EncodeFile(file1.c_str(), file2.c_str(), cpuCodec, abmodel);
						Helpers::DecodeFile(file2.c_str(), file3.c_str(), cpuCodec, abmodel);
						cout << "Comparing source and decoded files..." << endl;
						Helpers::CompareFiles(file1.c_str(), file3.c_str());
						continue;
					}
					else if (device == "g") {
						Helpers::EncodeFile(file1.c_str(), file2.c_str(), gpuCodec, abmodel);
						Helpers::DecodeFile(file2.c_str(), file3.c_str(), cpuCodec, abmodel);
						cout << "Comparing source and decoded files..." << endl;
						Helpers::CompareFiles(file1.c_str(), file3.c_str());
						continue;
					}
					else if (device == "cg") {
						string cpuFile = file2 + ".cpu";
						string gpuFile = file2 + ".gpu";
						string cpuDecompressed = cpuFile + ".decompressed";
						string gpuDecompressed = gpuFile + ".decompressed";

						cout << "CPU" << endl;
						Helpers::EncodeFile(file1.c_str(), cpuFile.c_str(), cpuCodec, abmodel);
						Helpers::DecodeFile(cpuFile.c_str(), cpuDecompressed.c_str(), cpuCodec, abmodel);
						cout << "Comparing source and decoded files..." << endl;
						Helpers::CompareFiles(file1.c_str(), cpuDecompressed.c_str());

						cout << "GPU" << endl;
						Helpers::EncodeFile(file1.c_str(), gpuFile.c_str(), gpuCodec, abmodel);
						Helpers::DecodeFile(gpuFile.c_str(), gpuDecompressed.c_str(), cpuCodec, abmodel);
						cout << "Comparing source and decoded files..." << endl;
						Helpers::CompareFiles(file1.c_str(), gpuDecompressed.c_str());
						continue;
					}
				}
				if (codecMode == "ca") {
					if (device == "c") {
						Helpers::EncodeFile(file1.c_str(), file2.c_str(), cpuCodec, camodel);
						Helpers::DecodeFile(file2.c_str(), file3.c_str(), cpuCodec, camodel);
						cout << "Comparing source and decoded files..." << endl;
						Helpers::CompareFiles(file1.c_str(), file3.c_str());
						continue;
					}
					else if (device == "g") {
						Helpers::EncodeFile(file1.c_str(), file2.c_str(), gpuCodec, camodel);
						Helpers::DecodeFile(file2.c_str(), file3.c_str(), cpuCodec, camodel);
						cout << "Comparing source and decoded files..." << endl;
						Helpers::CompareFiles(file1.c_str(), file3.c_str());
						continue;
					}
					else if (device == "cg") {
						string cpuFile = file2 + ".cpu";
						string gpuFile = file2 + ".gpu";
						string cpuDecompressed = cpuFile + ".decompressed";
						string gpuDecompressed = gpuFile + ".decompressed";

						cout << "CPU" << endl;
						Helpers::EncodeFile(file1.c_str(), cpuFile.c_str(), cpuCodec, camodel);
						Helpers::DecodeFile(cpuFile.c_str(), cpuDecompressed.c_str(), cpuCodec, camodel);
						cout << "Comparing source and decoded files..." << endl;
						Helpers::CompareFiles(file1.c_str(), cpuDecompressed.c_str());

						cout << "GPU" << endl;
						Helpers::EncodeFile(file1.c_str(), gpuFile.c_str(), gpuCodec, camodel);
						Helpers::DecodeFile(gpuFile.c_str(), gpuDecompressed.c_str(), cpuCodec, camodel);
						cout << "Comparing source and decoded files..." << endl;
						Helpers::CompareFiles(file1.c_str(), gpuDecompressed.c_str());
						continue;
					}
				}
			}
			if (command == "d") {
				if (file2 == "d") {
					file2 = file1 + ".decompressed";
				}

				if (codecMode == "s") {
					float prob;
					cout << "Enter exact zero probability with which file was encoded: " << std::endl;
					cin >> prob;
					smodel.SetZeroProbability(prob);
					if (device == "c") {
						Helpers::DecodeFile(file1.c_str(), file2.c_str(), cpuCodec, smodel);
						continue;
					}
				}
				if (codecMode == "a") {
					if (device == "c") {
						Helpers::DecodeFile(file1.c_str(), file2.c_str(), cpuCodec, amodel);
						continue;
					}
				}
				if (codecMode == "ab") {
					if (device == "c") {
						Helpers::DecodeFile(file1.c_str(), file2.c_str(), cpuCodec, abmodel);
						continue;
					}
				}
				if (codecMode == "ca") {
					if (device == "c") {
						Helpers::DecodeFile(file1.c_str(), file2.c_str(), cpuCodec, camodel);
						continue;
					}
				}
			}
			if (command == "q") {
				break;
			}
			cout << "Wrong arguments. Try again." << endl;
		}
		catch (std::exception& e)
		{
			cout << e.what() << endl;
		}
	}
	return 0;
}