#include <cmath>
#include <algorithm>
#include <limits>
// Exact branch evolution, with analytic transversal threshold events.
static bool segment(double &u,double &v,double T,double k,double a,int &nc) {
  if(T<0){v=-v; bool ok=segment(u,v,-T,k,a,nc);v=-v;return ok;}
  double r=std::sqrt(k), left=T;
  int b = u>a ? 1 : 0;
  if(std::abs(u-a)<2e-14){ if(std::abs(v)<1e-13)return false;b=v>0?1:0; }
  for(int it=0;left>1e-14 && it<1000;it++){
    double d=u-b,A=d+r*v,B=-2*(a-b),C=d-r*v;
    double next=std::numeric_limits<double>::infinity();
    auto consider=[&](double z){if(z>1+2e-13 && std::isfinite(z)){double t=std::log(z)/r;if(t<next)next=t;}};
    double disc=B*B-4*A*C;
    if(std::abs(A)<1e-15){if(std::abs(B)>1e-15)consider(-C/B);}
    else if(disc>=0){double q=-.5*(B+std::copysign(std::sqrt(disc),B));consider(q/A);if(q!=0)consider(C/q);}
    bool hit=next<=left+2e-13;
    double dt=hit?std::min(next,left):left;
    double ch=std::cosh(r*dt),sh=std::sinh(r*dt);
    double un=b+d*ch+r*v*sh,vn=d*sh/r+v*ch;
    u=un;v=vn;left-=dt;
    if(hit){u=a;nc++; if(std::abs(v)<1e-13)return false;b=1-b;}
    if(!std::isfinite(u)||!std::isfinite(v))return false;
  }
  return left<=1e-14;
}
extern "C" {
int flow(double *y,double T,double k,double a){int n=0;bool ok=segment(y[0],y[1],T,k,a,n);return ok?n:-1;}
int period(double *y,double l1,double l2,double al,double a,int direction){
  int n=0; bool ok=segment(y[0],y[1],direction*l1/2,1,a,n);
  if(ok)ok=segment(y[0],y[1],direction*l2,al,a,n);
  if(ok)ok=segment(y[0],y[1],direction*l1/2,1,a,n);
  return ok?n:-1;
}
void iterate_batch(double *ys,int N,int steps,double l1,double l2,double al,double a,int *counts){
  for(int i=0;i<N;i++)for(int j=0;j<steps;j++){
    double *y=ys+2*i;
    if(!std::isfinite(y[0]))break;
    // Outside [0,1] and moving outward: no return is possible.
    if((y[0]<-5 && y[1]<0)||(y[0]>6 && y[1]>0)){y[0]=y[1]=NAN;break;}
    int n=period(y,l1,l2,al,a,1);if(n<0){y[0]=y[1]=NAN;break;}counts[i]+=n;
  }
}
}
