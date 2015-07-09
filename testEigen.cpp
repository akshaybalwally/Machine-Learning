#include "eigen.h"
int main()
{
	eigen * neural = new eigen(5, 2, 100);
	neural->populate();
	neural->initializeThetas();
	neural->backPropogate(50);
	neural->test(20);

	return 0;

}