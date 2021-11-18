#include <vector>
#include <math.h>
#include <stdio.h>
#include <string>
#include <sstream>

#include <cxxopts.hpp>

void cubicSpline(double *W, double dWdx[3], double *dWdr, double dx[3], double sml);

void cubicSpline(double *W, double dWdx[3], double *dWdr, double dx[3], double sml) {

    double r, q, f;
    r = 0;
    for (int d = 0; d < 3; d++) {
        r += dx[d] * dx[d];
        dWdx[d] = 0;
    }
    r = sqrt(r);
    *dWdr = 0;
    *W = 0;
    q = r/sml;

    f = 8./M_PI * 1./(sml * sml * sml);

    if (q > 1) {
        *W = 0;
        *dWdr = 0.0;
    } else if (q > 0.5) {
        *W = 2. * f * (1.-q) * (1.-q) * (1-q);
        *dWdr = -6. * f * 1./sml * (1.-q) * (1.-q);
    } else if (q <= 0.5) {
        *W = f * (6. * q * q * q - 6. * q * q + 1.);
        *dWdr = 6. * f/sml * (3 * q * q - 2 * q);
    }
    for (int d = 0; d < 3; d++) {
        dWdx[d] = *dWdr/r * dx[d];
    }
}

int main(int argc, char *argv[]) {

    cxxopts::Options options("Sedov", "Generate initial distribution");

    bool asAscii = false;

    options.add_options()
            ("M,mass", "total system mass", cxxopts::value<double>()->default_value("1.9891e30")) //1.9891e30
            ("d,delta", "distance between mesh points", cxxopts::value<double>()->default_value("0.02625")) // 0.05625
            ("r,rmax", "maximal diameter", cxxopts::value<double>()->default_value("1"))
            ("o,output", "File name", cxxopts::value<std::string>()->default_value("sedov_kernel"))
            ("a,ascii", "generate ascii output", cxxopts::value<bool>(asAscii)) //->default_value("true")->implicit_value("true"))
            ("h,help", "Show this help");


    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    double MASS = result["mass"].as<double>(); //1.9891e30;
    double delta = result["delta"].as<double>(); //1.8e13;
    double R_MAX = result["rmax"].as<double>(); //3.2e14;
    double SEDOV_BLAST_RADIUS = R_MAX * 0.06; //R_MAX * 0.2;
    double rho = 1.;
    double sedovBlastEnergy = 1.0; //1e5; //2.e-7; // ??? //100.;
    double e0 = 1e-8;

    std::string outputBase = result["output"].as<std::string>();

    FILE *output;

    std::vector<std::vector<double>> x_vec, v_vec;
    std::vector<double> mass_vec, u_vec;
    std::vector<int> matId_vec;

    int particlesBlast = 0;
    double massBast = 0.;

    double e, f, g;
    double tmp, tmp1, tmp2, tmp3;
    double _x, _y, _z;
    double vx, vy, vz;

    bool write = false;
    double r_max = R_MAX * 1.01;

    double W;
    double dWdx[3];
    double dWdr;
    double dr[3];

    _x = _y = _z = 0.0;

    e = f = g = -r_max;

    int n = 0;
    while (e < r_max) {

        f = -r_max;
        g = -r_max;

        while (f < r_max) {
            g = -r_max;

            while (g < r_max) {
                tmp1 = (e - _x) * (e - _x);
                tmp2 = (f - _y) * (f - _y);
                tmp3 = (g - _z) * (g - _z);
                tmp = tmp1 + tmp2 + tmp3;
                tmp = sqrt(tmp);

                if (tmp < R_MAX) {
                    write = true;
                }

                if (tmp < SEDOV_BLAST_RADIUS) {
                    particlesBlast++;
                }
                if (write) {
                    n++;
                    write = false;
                }
                g += delta;
            }
            f += delta;
        }
        e += delta;

    }

    double m = (rho * ((4./3.) * M_PI * pow(R_MAX,3)))/n;
    double sml = 5 * delta;
    double polyGamma = 1.4;
    double uBlast = sedovBlastEnergy / particlesBlast;
    double u0 = e0;
    double u;
    double uBlastTest = sedovBlastEnergy / (((4./3.) * M_PI * pow(SEDOV_BLAST_RADIUS,3)) * particlesBlast);

    double uBlastSum = 0;


    e = f = g = -r_max;

    printf("numParticles: %i, mass of one particle: %e\n", n, m);
    printf("numParticles blast: %i\n", particlesBlast);

    std::stringstream asciiOutput;
    asciiOutput << outputBase << ".output";
    std::string _asciiOutput = asciiOutput.str();

    output = fopen(_asciiOutput.c_str(), "w");

    while (e < r_max) {
        f = -r_max;
        g = -r_max;
        while (f < r_max) {
            g = -r_max;
            while (g < r_max) {
                tmp1 = (e - _x) * (e - _x);
                tmp2 = (f - _y) * (f - _y);
                tmp3 = (g - _z) * (g - _z);
                tmp = tmp1 + tmp2 + tmp3;
                tmp = sqrt(tmp);

                //if (tmp < R_MAX)
                //    write = true;
                //if (write) {

                if (tmp < R_MAX) {
                    n++;
                    vx = vy = vz = 0.0;
                    u = u0;

                    if (tmp < SEDOV_BLAST_RADIUS) {
                        dr[0] = e;
                        dr[1] = f;
                        dr[2] = g;
                        cubicSpline(&W, dWdx, &dWdr, dr, 0.029);
                        u += W; //uBlast;
//                        printf("tmp = %e u += uBlast = %e W = %e dWdx = (%e, %e, %e), dWdr = %e\n", tmp, u, W, dWdx[0], dWdx[1], dWdx[2], dWdr);
                        uBlastSum += W;
                    }

                    // input file format for file <string.XXXX>:
                    // 1:x[0] 2:x[1] 3:x[2] 4:v[0] 5:v[1] 6:v[2] 7:mass 8:energy 9:material type
                    fprintf(output, "%e %e %e 0.0 0.0 0.0 %e %e 0\n", e, f, g, m, u);
                }

                //write = false;

                g += delta;
            }
            f += delta;
        }
        e += delta;
    }

    fclose(output);
    std::cout << "[ASCII] Output written to: " << _asciiOutput << std::endl;


    //printf("polyK: %e\n", polyK);
    printf("u0: %e, uBlast: %e\n", u0, uBlast);
    printf("uBlastSum: %e\n", uBlastSum);


    return 0;
}

