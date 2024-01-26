#include <stdio.h>
#include <stdbool.h>
#include <math.h>

int main()
{
    int i,j,m,n,temp;
    printf("Enter dimensions m*n : ");
    scanf("%d %d",&m,&n);
    int matrix[m][n];

    for(i=0;i<m;i++)
    {
        for(j=0;j<n;j++)
        {
            scanf("%d",&matrix[i][j]);
        }
    }
    printf("\n");
    for(i=0;i<m;i++)
    {
        for(j=0;j<n;j++)
        {
            printf("%d ",matrix[i][j]);
        }
        printf("\n\r");
    }
    for(i=0;i<m;i++)
    {
        for(j=i;j<n;j++)
        {
            if(i == j)
            {
                continue;
            }
            else
            {
                temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
    }
    printf("\n\r");
    for(i=0;i<n;i++)
    {
        for(j=0;j<m;j++)
        {
            printf("%d ",matrix[i][j]);
        }
        printf("\n\r");
    }
    return 0;
}