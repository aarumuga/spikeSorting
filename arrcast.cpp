#include <iostream>
#include <stdio.h>
#include <limits>
#include <vector>

using namespace std;

struct test_s {
    public:
    vector<int> v;
};

int main(void) {
    int *temp = new int(6);
    for (int i = 0; i < 6; i++) {
        temp[i] = i;
    }
    for (int i = 0; i < 2; i++) {
         for (int j = 0; j < 3 ; j++) {
             std::cout <<((int(*)[3])temp)[i][j]<< std::endl;
         }
    }
    return 0;
}
