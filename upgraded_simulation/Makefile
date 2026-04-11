CXX = g++
CXXFLAGS = -O3 -Wall -shared -std=c++11 -fPIC

TARGET = libpercolation.so

all: $(TARGET)

$(TARGET): percolation.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) percolation.cpp

clean:
	rm -f $(TARGET)
