#include "colors.inc"
#include "textures.inc"
#include "stones1.inc"


#declare Lightbulb = union {
    merge {
      sphere { <0,0,0>,1 }
      cylinder {
        <0,0,1>, <0,0,0>, 1
        scale <0.35, 0.35, 1.0>
        translate  0.5*z
      }
      texture {
        pigment {color rgb <1, 1, 1>}
        finish {ambient .8 diffuse .6}
      }
    }
    cylinder {
      <0,0,1>, <0,0,0>, 1
      scale <0.4, 0.4, 0.5>
      texture { Brass_Texture }
      translate  1.5*z
    }
    rotate -90*x
    scale .5
  }
camera
{
  location <10, 10, 0.1>
  look_at <0, 0, 0>
  up <0,1,0>
  right <1,0,0>
}


  light_source {
    <10, 10, 3>
    color Yellow
    area_light <1, 0, 0>, <0, 1, 0>, 2, 2
    adaptive 1
    jitter
  }


#declare Texture_W =
 texture{ pigment{ color White*0.9}
          normal { bumps 1 scale 0.025}
          finish { diffuse 0.9 specular 1}
        } // end of texture
#declare Texture_S =
 texture{ T_Stone10 scale 1
          normal { agate 0.5 scale 0.25}
          finish { diffuse 0.9 phong 1 }
        } // end of texture
//------------------------------------------------------
//sphere { <0,0,0>, 1
//         texture{ crackle  scale 1.5 turbulence 0.1
//           texture_map {[0.00 Texture_W]
//                        [0.05 Texture_W]
//                        [0.05 Texture_S]
//                        [1.00 Texture_S]
//                       }// end of texture_map
//                   scale 0.2
//         } // end of texture ---------------------------
//  scale<1,1,1>  rotate<0,0,0>  translate<0.40,1,0>
//}  // end of sphere ------------------------------------


#declare Pigment_1 =   //------------------
pigment{ crackle
         turbulence 0.35 scale 0.005
         color_map{
          [0.00 color rgb<1,1,1>*0]
          [0.08 color rgb<1,1,1>*0]
          [0.40 color rgb<1,0.55,0>]
          [1.00 color rgb<1,1,0.8>]
         } // end of color_map
} // end of pigment -----------------------
//object{ //---------------------------------
 // Round_Box(<-1,0,-1.25>,<2,2,2>,0.15,0)
  //texture{ pigment{ Pigment_1  }
   //        finish { phong 1 }
    //     } // end texture
//} // --------------------------------------
