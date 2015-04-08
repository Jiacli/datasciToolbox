#include <iostream>
#include <map>
#include <vector>
#include <cstdio>
#include <cstdlib>
using namespace std;

map<int,int> tp,fp,fn,classes;
vector< pair<int,int> > preds;


double get_prec( int tp, int fp , int fn ){
  if ( tp + fp > 0 )
    return tp*1.0/ ( tp + fp );
  return 0;
}

double get_rec( int tp, int fp, int fn ) {
  if ( tp + fn > 0 )
    return tp*1.0/ ( tp + fn );
  return 0;
}

double get_f1( int tp, int fp, int fn ) {
  double p = get_prec(tp,fp,fn);
  double r = get_rec(tp,fp,fn);
	
  if ( p + r > 0 ) 	
    return (2.0*p*r/(p+r));
  return 0;
}

int main( int argc , char *argv[] ){
  if ( argc != 2 ) {
    puts("./eval file-name. \n\n File-name consists of two columns, predicted-labels and true-labels.\n");
    exit(0);
  }
	
  FILE *f = fopen( argv[1] , "r");
  if ( f == NULL ) {
    puts("Unable to open input file");
    exit(0);	
  }
	
  int a,b;
  while( fscanf(f,"%d %d",&a,&b) == 2 ) {
    preds.push_back( make_pair(a,b) );
    classes[b] = 1;	
  }

  int ta = 0, tb = 0;
  for( int i = 0; i < int(preds.size()); ++i ) {
    int a = preds[i].first, b = preds[i].second;

    if ( classes.count(a) == 0 ) {
      printf("Undefined class %d (In line %d)\n", a , i+1 );
      exit(0);
    }

    if ( a == b ) 
      tp[a]++, ta++;
    else 
      fp[a]++,  fn[b]++, tb++;
  }

  int sumTP = 0, sumFP = 0, sumFN = 0;
  double sumPrec = 0, sumRec = 0, sumF1  = 0;

  printf("%10s %5s %5s %5s %15s %15s %15s\n","CLASS","TP","FP","FN","PREC","REC","F1");
  for ( map<int,int>::iterator it = classes.begin(); 
	it != classes.end(); it++ ){
	
    int cl = (it->first);
    int _tp = tp[cl], _fp = fp[cl], _fn = fn[cl];
    double prec = get_prec(_tp,_fp,_fn), rec = get_rec(_tp,_fp,_fn),
      f1 = get_f1(_tp,_fp,_fn);
		
    printf("%10d %5d %5d %5d %15.7lf %15.7lf %15.7lf\n",cl,_tp,_fp,_fn,prec,rec,f1);
    sumTP += _tp , sumFN += _fn, sumFP += _fp;
    sumPrec += prec, sumRec += rec, sumF1 += f1;
  }
	
  double P = get_prec(sumTP,sumFP,sumFN), R = get_rec(sumTP,sumFP,sumFN),
    F1 = get_f1(sumTP,sumFP,sumFN);

  int cnt = max( 1 , int(classes.size()) );
  double avgP = sumPrec/cnt, avgR = sumRec/cnt, avgF1 = sumF1/cnt;
	
  printf("%10s %5d %5d %5d %15.7lf %15.7lf %15.7lf\n","MICRO-AVG",sumTP,sumFP,sumFN,P,R,F1);
  printf("%10s %5s %5s %5s %15.7lf %15.7lf %15.7lf\n","MACRO-AVG","","","",avgP,avgR,avgF1);
}
