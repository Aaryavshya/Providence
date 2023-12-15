#include <stdio.h>
#include <math.h>
/*final*/
int main()
{
    int i,j,id_1,id_2,n,m,p,l_2,l_1,t_1;
    id_2 = 0;
    printf("Declare the dimensions of the matrices as n*m X m*p: n,m,p: ");
    scanf("%d %d %d",&n,&m,&p);
    float m_1[n][m];
    float m_2[m][p];
    float r[n][p];
    printf("{\n\r");
    for(i=0;i<n;i++)
    {
        for(j=0;j<m;j++)
        {
            scanf("%f",&m_1[i][j]);
        }
    }
    printf("}\n\r");
    printf("Second matrix\n\r");
    printf("{\n\r");
    for(i=0;i<m;i++)
    {
        for(j=0;j<p;j++)
        {
            scanf("%f",&m_2[i][j]);
        }
    }
    for(i=0;i<n;i++)
    {
        for(j=0;j<p;j++)
        {
            r[i][j]=0;
            for(l_1=0;l_1<m;l_1++)
            {
                id_1=m_1[i][l_1]*m_2[l_1][j];
                r[i][j]=r[i][j]+id_1;
            }
        }
    }
    printf("}");
    printf("\n\r");
    for(i=0;i<n;i++)
    {
        for(j=0;j<p;j++)
        {
            printf("%f ",r[i][j]);
        }
        printf("\n\r");
    }
    return 0;
}