 
#include <stdio.h>
 
int main()
{
    float sum=0;
    float a=2,b=1;
    for(int i=1;i<=20;i++)
    {
        int tmp = a;
        printf("b --->>%f\n",b);
        a+=b;
        b = tmp;
        sum+=a/b;
    }
    printf("%9.6f\n",sum);  
}
