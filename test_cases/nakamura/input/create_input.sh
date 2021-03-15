gcc projectile.c -lm
./a.out > projectile.0000
gcc target.c -lm
./a.out > target.0000

gcc  weibullit.c -o weibullit -lm 

./weibullit -v -n `wc -l target.0000` -k 5e34 -m 8.5 -f target.0000 -o targetw.0000 -t 0 -M 28
#./weibullit -v -n `wc -l target.0000` -k 5e34 -m 8 -f target.0000 -o targetw.0000 -t 0 -M 28
#./weibullit -v -n `wc -l projectile.0000` -k 4e35 -m 8.5 -f projectile.0000 -o projectilew.0000 -t 1


cat projectile.0000 targetw.0000  > impact.0000
