#include <iostream>
using namespace std;
#include<bits/stdc++.h> 

int lis(char* s1, char* s2, int num1, int num2){
   int l[num1+1][num2+1];
   for(int i=0;i<=num1;i++){
   		for(int j=0;j<=num2;j++){
   			if(i==0)
			   l[i][j] = j;
			else if(j==0)
				l[i][j] = i;
			else if(s1[i-1] == s2[j-1])
				l[i][j] = l[i-1][j-1];
			else
				l[i][j] =1+ min(min (l[i-1][j-1],l[i-1][j]),l[i][j-1])	;
		}
   }
   
   return l[num1][num2];
}

int min(int a, int b){
	return a<b?a:b;
}

int main() {
	int test;
	cin>>test;
	int ans=1;
	while(test--){
	   int num1;int num2;
	    cin>>num1;
	    cin>>num2;
	    char s1[num1]; char s2[num2];
	    for(int i=0;i<num1;i++)
	        cin>>s1[i];
	    for(int i=0;i<num2;i++)
	        cin>>s2[i];
	   
	    ans = lis(s1,s2,num1,num2);
	    cout<<ans<<endl;
	}
	return 0;
}

