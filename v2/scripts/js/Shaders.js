var RefractionShader = function(){

    this.uniforms = THREE.UniformsUtils.merge([
    { 
        "envMap": { type: "t", value: null },
        "map": { type: "t", value: null },
        "tex2": { type: "t", value: null },
        "alpha": { type: "t", value: null },
        "refractionMap": { type: "t", value: null },
        "flipEnvMap": { type: "f", value: 1.0 },
        "time": { type: "f", value: 0.0 },
        "noiseScale":{type: "f", value:0.0},
        "noiseDetail":{type: "f", value:0.0},
        "refractionRatio":{type: "f", value:0.95},
        "diffuse":{type: "v3", value:new THREE.Vector3(1.0,1.0,1.0)},
        "reflectivity":{type: "f", value:1.0},
        "mouse":{type: "v2", value:null}
      }
  ]);

    this.vertexShader = [

            "varying vec3 vReflect;",
            "uniform float refractionRatio;",
            "uniform sampler2D map;",
            "uniform sampler2D refractionMap;",

            "varying vec2 vUv;",
            "uniform vec2 mouse;",

            "uniform float time;",

            "vec3 transformDirection( in vec3 normal, in mat4 matrix ) {",
            "   return normalize( ( matrix * vec4( normal, 0.0 ) ).xyz );",
            "}",
            "float luminance(vec3 c){",
            "    return dot(c, vec3(.2126, .7152, .0722));",
            "}",
            "void main(){",

            "   vUv = uv;",
            "   vec4 color = texture2D(map, vUv);",
            // "   float depth = smoothstep(0.499,0.501,( color.r + color.g + color.b ) / 3.0);",
            // "   float depth = ( color.r + color.g + color.b ) / 3.0;",
            "   float depth = luminance(color.rgb);",
            "   float z = ( depth*2.0 - 1.0 ) * (4500.0 - 800.0) + 800.0;",

            "   vec3 pos = vec3(position.x, position.y, z*0.2);",
            // "   vec3 pos = vec3(position.x, position.y, sin(time+uv.x)*z*0.1);",
            // "   vec3 pos = vec3(position.x, position.y, position.z + sin(time*10.0 + position.x*0.1)*100.0);",
            // "   vec3 pos = vec3(position.x, position.y, z*sin(time*10.0 + z)*0.1);",
            "   vec4 mvPosition = modelViewMatrix * vec4( pos, 1.0 );",
            "   gl_Position = projectionMatrix * mvPosition;",
            "       vec3 objectNormal = normal;",
            "   vec4 worldPosition = modelMatrix * vec4( pos, 1.0 );    ",
            "   vec3 worldNormal = transformDirection( objectNormal, modelMatrix );",
            "   vec3 cameraToVertex = normalize( worldPosition.xyz - cameraPosition );",
            "   vec4 rMap = texture2D(refractionMap, vUv);",
            "    vReflect = refract( cameraToVertex, worldNormal, refractionRatio );",
            // "    vReflect = refract( cameraToVertex, worldNormal, depth );",
            // "    vReflect = refract( cameraToVertex, worldNormal, dot(rMap.rgb, vec3(1.0)/3.0) );",
            // "    vReflect = refract( cameraToVertex, worldNormal, depth );",
            // "    vReflect = refract( cameraToVertex, worldNormal, mouse.x );",
            // "   vReflect = reflect( cameraToVertex, worldNormal );",

            "}"

    ].join("\n");

    this.fragmentShader = [

            "uniform float reflectivity;",
            "uniform samplerCube envMap;",
            "uniform sampler2D alpha;",
            "uniform sampler2D map;",
            "uniform sampler2D tex2;",
            "uniform sampler2D refractionMap;",
            "uniform float flipEnvMap;",
            "uniform float mixAmt;",

            "varying vec3 vReflect;",
            "varying vec2 vUv;",
            "uniform vec3 diffuse;",

            "float luminance(vec3 c){",
            "    return dot(c, vec3(.2126, .7152, .0722));",
            "}",
            "void main() {",
            "   vec3 outgoingLight = vec3( 0.0 );",
            "   vec4 diffuseColor = vec4( diffuse, 1.0 );",
            "   float flipNormal = ( -1.0 + 2.0 * float( gl_FrontFacing ) );",
            "   float specularStrength;",
            "   specularStrength = 1.0;",
            "   outgoingLight = diffuseColor.rgb;",

            "   vec3 reflectVec = vReflect;",

            "   vec4 envColor = textureCube( envMap, flipNormal * vec3( flipEnvMap * reflectVec.x, reflectVec.yz ) );",
            // "    vec4 envColor = texture2D( map, vUv );",
            // "    envColor.xyz = inputToLinear( envColor.xyz );",
            "   outgoingLight = mix( outgoingLight, outgoingLight * envColor.xyz, specularStrength * reflectivity );",

            "   gl_FragColor = vec4( outgoingLight, diffuseColor.a );",
            // "   gl_FragColor = texture2D(refractionMap, vUv);",
            // "   vec4 aMap = texture2D(map, vUv);",

            // "   gl_FragColor = mix(texture2D(tex2, vUv), texture2D(refractionMap, vUv), dot(aMap.rgb, vec3(1.0))/3.0);",
            // "   vec4 rMap = texture2D(refractionMap, vUv);",

            // "   vec4 col = vec4( outgoingLight, diffuseColor.a );",
            // "   gl_FragColor = mix(rMap,  dot(rMap.rgb, vec3(1.0)));",
            // "   gl_FragColor = vec4( 1.0, 0.0,0.0,1.0 );",
            // "   vec4 alphaTex = texture2D(alpha, vUv);",
            // "   vec4 s = texture2D(map2, vUv);",
            // "   vec4 t = vec4( outgoingLight, diffuseColor.a );",
            // "   vec4 col = mix(t, s, dot(alpha.rgb, vec3(1.0)/3.0));",
            // "   vec4 col = mix(s, alpha,  dot(s.rgb, vec3(1.0)/3.0));",
            // "   vec4 col = mix(alpha, s,  dot(luminance(s.rgb), 1.0/3.0));",
            // "   vec4 col = mix(t, vec4(1.0,0.0,0.0,1.0), dot(alphaTex.rgb, vec3(1.0)/3.0));",
            // "   vec4 col = mix(s, t, mixAmt);",
            // "    gl_FragColor = textureCube( envMap, flipNormal * vec3( flipEnvMap * reflectVec.x, reflectVec.yz ) );",

            "}"

        ].join("\n")
}
var AShader = function(){
        this.uniforms = THREE.UniformsUtils.merge([
            {
                "BUF_A"  : { type: "t", value: null },
                "BUF_B"  : { type: "t", value: null },
                "alpha"  : { type: "t", value: null },
                "FRAME"  : { type: "f", value: null },
                "resolution"  : { type: "v2", value: null },
                "mouse"  : { type: "v2", value: null },
                "time"  : { type: "f", value: null },
            }
        ]);
        this.vertexShader = [

            "varying vec2 vUv;",
            "void main() {",
            "    vUv = uv;",
            "    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );",
            "}"
        
        ].join("\n");
        
        this.fragmentShader = [
            
            "uniform sampler2D BUF_A;",
            "uniform sampler2D BUF_B;",
            "uniform sampler2D alpha;",
            "uniform float FRAME;",
            "uniform vec2 resolution;",
            "uniform vec2 mouse;",
            "uniform float time;",
            "varying vec2 vUv;",
            "void main()",
            "{",
            "    const float _K0 = -20.0/6.0; // center weight",
            "    const float _K1 = 4.0/6.0; // edge-neighbors",
            "    const float _K2 = 1.0/6.0; // vertex-neighbors",
            "    const float cs = -0.5; // curl scale",
            "    const float ls = 0.03; // laplacian scale",
            "    const float ps = -0.05; // laplacian of divergence scale",
            "    const float ds = 0.05; // divergence scale",
            "    const float is = 0.01; // image derivative scale",
            "    const float pwr = 1.0; // power when deriving rotation angle from curl",
            "    const float amp = 1.0; // self-amplification",
            "    const float sq2 = 0.7; // diagonal weight",

            // "    vec2 texel = mouse.y*10.0 / resolution.xy;",
            "    vec2 texel = 1.0 / resolution.xy;",
            "    ",
            "    // 3x3 neighborhood coordinates",
            "    float step_x = texel.x;",
            "    float step_y = texel.y;",
            "    vec2 n  = vec2(0.0, step_y);",
            "    vec2 ne = vec2(step_x, step_y);",
            "    vec2 e  = vec2(step_x, 0.0);",
            "    vec2 se = vec2(step_x, -step_y);",
            "    vec2 s  = vec2(0.0, -step_y);",
            "    vec2 sw = vec2(-step_x, -step_y);",
            "    vec2 w  = vec2(-step_x, 0.0);",
            "    vec2 nw = vec2(-step_x, step_y);",
            "    ",
            "    // sobel filter",
            "    vec3 im = texture2D(BUF_B, vUv).xyz;",
            "    vec3 im_n = texture2D(BUF_B, vUv+n).xyz;",
            "    vec3 im_e = texture2D(BUF_B, vUv+e).xyz;",
            "    vec3 im_s = texture2D(BUF_B, vUv+s).xyz;",
            "    vec3 im_w = texture2D(BUF_B, vUv+w).xyz;",
            "    vec3 im_nw = texture2D(BUF_B, vUv+nw).xyz;",
            "    vec3 im_sw = texture2D(BUF_B, vUv+sw).xyz;",
            "    vec3 im_ne = texture2D(BUF_B, vUv+ne).xyz;",
            "    vec3 im_se = texture2D(BUF_B, vUv+se).xyz;",

            "    float dx = 3.0 * (length(im_e) - length(im_w)) + (length(im_ne) + length(im_se) - length(im_sw) - length(im_nw));",
            "    float dy = 3.0 * (length(im_n) - length(im_s)) + (length(im_nw) + length(im_ne) - length(im_se) - length(im_sw));",

            "    // vector field neighbors",
            "    vec3 uv =    texture2D(BUF_A, vUv).xyz;",
            "    vec3 uv_n =  texture2D(BUF_A, vUv+n).xyz;",
            "    vec3 uv_e =  texture2D(BUF_A, vUv+e).xyz;",
            "    vec3 uv_s =  texture2D(BUF_A, vUv+s).xyz;",
            "    vec3 uv_w =  texture2D(BUF_A, vUv+w).xyz;",
            "    vec3 uv_nw = texture2D(BUF_A, vUv+nw).xyz;",
            "    vec3 uv_sw = texture2D(BUF_A, vUv+sw).xyz;",
            "    vec3 uv_ne = texture2D(BUF_A, vUv+ne).xyz;",
            "    vec3 uv_se = texture2D(BUF_A, vUv+se).xyz;",
            "    ",
            "    // uv.x and uv.y are our x and y components, uv.z is divergence ",

            "    // laplacian of all components",
            "    vec3 lapl  = _K0*uv + _K1*(uv_n + uv_e + uv_w + uv_s) + _K2*(uv_nw + uv_sw + uv_ne + uv_se);",
            "    float sp = ps * lapl.z;",
            "    ",
            "    // calculate curl",
            "    // vectors point clockwise about the center point",
            "    float curl = uv_n.x - uv_s.x - uv_e.y + uv_w.y + sq2 * (uv_nw.x + uv_nw.y + uv_ne.x - uv_ne.y + uv_sw.y - uv_sw.x - uv_se.y - uv_se.x);",
            "    ",
            "    // compute angle of rotation from curl",
            "    float sc = cs * sign(curl) * pow(abs(curl), pwr);",
            "    ",
            "    // calculate divergence",
            "    // vectors point inwards towards the center point",
            "    float div  = uv_s.y - uv_n.y - uv_e.x + uv_w.x + sq2 * (uv_nw.x - uv_nw.y - uv_ne.x - uv_ne.y + uv_sw.x + uv_sw.y + uv_se.y - uv_se.x);",
            "    float sd = ds * div;",

            "    vec2 norm = normalize(uv.xy);",
            "    ",
            "    // temp values for the update rule",
            "    float ta = amp * uv.x + ls * lapl.x + norm.x * sp + uv.x * sd + is * dx;",
            "    float tb = amp * uv.y + ls * lapl.y + norm.y * sp + uv.y * sd + is * dy;",

            "    // rotate",
            "    float a = ta * cos(sc) - tb * sin(sc);",
            "    float b = ta * sin(sc) + tb * cos(sc);",
            "    ",
            "    // initialize with noise",
            // "    if(FRAME<10.0) {",
            // "        gl_FragColor = -0.5 + texture2D(BUF_B, vUv);",
            // "    } else {",
            // "        gl_FragColor = clamp(vec4(a,b,div,1), -1., 1.);",
            // "    }",
            "    vec4 col = clamp(vec4(a,b,div,1), -1., 1.);",
            "    vec4 inputTex = -0.5 + texture2D(BUF_B, vUv);",
            "    vec4 alpha = texture2D(alpha, vUv);",
            "    // initialize with image",
            "    if(FRAME<10.0) {",
            "        gl_FragColor = inputTex;",
            "    } else {",
            // "        gl_FragColor = vec4(clamp(im + ds * diffusion_im, 0.0, 1.0), 0.0);",
            "        gl_FragColor = col;",
            // "        gl_FragColor = mix(inputTex, col, dot(alpha.rgb, vec3(1.0))/3.0);",
            "    }",
            "    ",

            "}",
        
        ].join("\n");
}
var BShader = function(){
        this.uniforms = THREE.UniformsUtils.merge([
            {
                "INPUT"  : { type: "t", value: null },
                "BUF_A"  : { type: "t", value: null },
                "BUF_B"  : { type: "t", value: null },
                "alpha"  : { type: "t", value: null },
                "FRAME"  : { type: "f", value: null },
                "resolution"  : { type: "v2", value: null },
                "mouse"  : { type: "v2", value: null },
                "time"  : { type: "f", value: null },
            }
        ]);
        this.vertexShader = [

            "varying vec2 vUv;",
            "void main() {",
            "    vUv = uv;",
            "    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );",
            "}"
        
        ].join("\n");
        
        this.fragmentShader = [
            
            "uniform sampler2D INPUT;",
            "uniform sampler2D BUF_A;",
            "uniform sampler2D BUF_B;",
            "uniform sampler2D alpha;",
            "uniform float FRAME;",
            "uniform vec2 resolution;",
            "uniform vec2 mouse;",
            "uniform float time;",
            "varying vec2 vUv;",
            "void main()",
            "{",
            "    const float ds = 0.2; // diffusion rate",
            "    const float darken = 0.0; // darkening",
            "    const float D1 = 0.2;  // edge neighbors",
            "    const float D2 = 0.05; // vertex neighbors",
            "    ",
            // "    vec2 texel = mouse.y*10.0 / resolution.xy;",
            "    vec2 texel = 1.0 / resolution.xy;",
            "    ",
            "    // 3x3 neighborhood coordinates",
            "    float step_x = texel.x;",
            "    float step_y = texel.y;",
            "    vec2 n  = vec2(0.0, step_y);",
            "    vec2 ne = vec2(step_x, step_y);",
            "    vec2 e  = vec2(step_x, 0.0);",
            "    vec2 se = vec2(step_x, -step_y);",
            "    vec2 s  = vec2(0.0, -step_y);",
            "    vec2 sw = vec2(-step_x, -step_y);",
            "    vec2 w  = vec2(-step_x, 0.0);",
            "    vec2 nw = vec2(-step_x, step_y);",
            "    ",
            "    vec3 components = texture2D(BUF_A, vUv).xyz;",
            "    ",
            "    float a = components.x;",
            "    float b = components.y;",
            "    ",
            "    vec3 im =    texture2D(BUF_B, vec2(mouse)*0.001+vUv).xyz;",
            "    vec3 im_n =  texture2D(BUF_B, vec2(mouse)*0.001+vUv+n).xyz;",
            "    vec3 im_e =  texture2D(BUF_B, vec2(mouse)*0.001+vUv+e).xyz;",
            "    vec3 im_s =  texture2D(BUF_B, vec2(mouse)*0.001+vUv+s).xyz;",
            "    vec3 im_w =  texture2D(BUF_B, vec2(mouse)*0.001+vUv+w).xyz;",
            "    vec3 im_nw = texture2D(BUF_B, vec2(mouse)*0.001+vUv+nw).xyz;",
            "    vec3 im_sw = texture2D(BUF_B, vec2(mouse)*0.001+vUv+sw).xyz;",
            "    vec3 im_ne = texture2D(BUF_B, vec2(mouse)*0.001+vUv+ne).xyz;",
            "    vec3 im_se = texture2D(BUF_B, vec2(mouse)*0.001+vUv+se).xyz;",

            "    float D1_e = D1 * a;",
            "    float D1_w = D1 * -a;",
            "    float D1_n = D1 * b;",
            "    float D1_s = D1 * -b;",
            "    float D2_ne = D2 * (b + a);",
            "    float D2_nw = D2 * (b - a);",
            "    float D2_se = D2 * (a - b);",
            "    float D2_sw = D2 * (- a - b);",

            "    vec3 diffusion_im = -darken * length(vec2(a, b)) * im + im_n*D1_n + im_ne*D2_ne + im_e*D1_e + im_se*D2_se + im_s*D1_s + im_sw*D2_sw + im_w*D1_w + im_nw*D2_nw;",

            // "    vec4 col = vec4(clamp(im + (mouse.x*1.0) * diffusion_im, 0.0, 1.0), 0.0);",
            "    vec4 col = vec4(clamp(im + ds * diffusion_im, 0.0, 1.0), 0.0);",
            "    vec4 inputTex = texture2D(INPUT, vUv);",
            "    vec4 alpha = texture2D(alpha, vUv);",
            "    // initialize with image",
            "    if(FRAME<10.0) {",
            "        gl_FragColor = inputTex;",
            "    } else {",
            // "        gl_FragColor = vec4(clamp(im + ds * diffusion_im, 0.0, 1.0), 0.0);",
            // "        gl_FragColor = mix(inputTex, col, dot(alpha.rgb, vec3(1.0))/3.0);",
            "        gl_FragColor = col;",
            "    }",
            "}",
        
        ].join("\n");
}
var OUTPUTShader = function(){
        this.uniforms = THREE.UniformsUtils.merge([
            {
                "BUF_B"  : { type: "t", value: null },
                "alpha"  : { type: "t", value: null },
                "mouse"  : { type: "v2", value: null },
                "resolution"  : { type: "v2", value: null },
                "time"  : { type: "f", value: null },
            }
        ]);
        this.vertexShader = [

            "varying vec2 vUv;",
            "void main() {",
            "    vUv = uv;",
            "    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );",
            "}"
        
        ].join("\n");
        
        this.fragmentShader = [
            
            "uniform sampler2D BUF_B;",
            "uniform vec2 resolution;",
            "uniform vec2 mouse;",
            "uniform float time;",
            "varying vec2 vUv;",
            
            "void main()",
            "{",
            "   vec3 col = texture2D(BUF_B, vUv).rgb;",
            "   gl_FragColor = vec4(col*vec3(1.0),1.0);",
            "}",
 
        ].join("\n");
}
var PassShader = function(){
        this.uniforms = THREE.UniformsUtils.merge([
            {
                "texture"  : { type: "t", value: null },
                "mouse"  : { type: "v2", value: null },
                "resolution"  : { type: "v2", value: null },
                "time"  : { type: "f", value: null },
            }
        ]);
        this.vertexShader = [

            "varying vec2 vUv;",
            "void main() {",
            "    vUv = uv;",
            "    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );",
            "}"
        
        ].join("\n");
        
        this.fragmentShader = [
            
            "uniform sampler2D texture;",
            "uniform vec2 resolution;",
            "uniform vec2 mouse;",
            "uniform float time;",
            "varying vec2 vUv;",
            
            "void main()",
            "{",
            "   vec3 col = texture2D(texture, vUv).rgb;",
            "   gl_FragColor = vec4(col,1.0);",
            "}",
 
        ].join("\n");
}
var DifferencingShader = function(){
    this.uniforms = THREE.UniformsUtils.merge( [

        {
            "texture"  : { type: "t", value: null },
            "mouse"  : { type: "v2", value: null },
            "resolution"  : { type: "v2", value: null },
            "time"  : { type: "f", value: null },
            "texture2"  : { type: "t", value: null },
        }
    ] ),

    this.vertexShader = [

        "varying vec2 vUv;",
        "void main() {",
        "    vUv = uv;",
        "    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );",
        "}"
    
    ].join("\n"),
    
    this.fragmentShader = [
        
        "uniform sampler2D texture;",
        "uniform sampler2D texture2;",
        "uniform vec2 resolution;",
        "uniform vec2 mouse;",
        "uniform float time;",
        "varying vec2 vUv;",

        "void main() {",
        "  vec4 tex0 = texture2D(texture, vUv);",
        "  vec4 tex1 = texture2D(texture2, vUv);",
        "  vec4 fc = (tex1 - tex0);",
        "  gl_FragColor = vec4(tex1);",
        "}"
    
    ].join("\n")
    
}

var FlowShader = function(){

    this.uniforms = THREE.UniformsUtils.merge( [

        {
            "texture"  : { type: "t", value: null },
            "mouse"  : { type: "v2", value: null },
            "resolution"  : { type: "v2", value: null },
            "time"  : { type: "f", value: null },
            "r2"  : { type: "f", value: null }

        }
    ] ),

    this.vertexShader = [

        "varying vec2 vUv;",
        "void main() {",
        "    vUv = uv;",
        "    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );",
        "}"
    
    ].join("\n"),
    
    this.fragmentShader = [
        
        "uniform vec2 resolution;",
        "uniform float time;",
        "uniform float r2;",
        "uniform sampler2D texture;",
        "varying vec2 vUv;",
        "uniform vec2 mouse;",

        "void main( void ){",
        "    vec2 uv = vUv;",

        "    vec2 e = 1.0/resolution.xy;",


        "    float am1 = 0.5 + 0.5*0.927180409;",
        "    float am2 = 10.0;",

        "    for( int i=0; i<20; i++ ){",
        "       float h  = dot( texture2D(texture, uv*0.5          ).xyz, vec3(1.0) );",
        "       float h1 = dot( texture2D(texture, uv+vec2(e.x,0.0)).xyz, vec3(1.0) );",
        "       float h2 = dot( texture2D(texture, uv+vec2(0.0,e.y)).xyz, vec3(1.0) );",
        "       vec2 g = 0.001*vec2( (h-h2), (h-h1) )/e;",
        // "        vec2 g = 0.001*vec2( (h1-h), (h2-h) )/e;",
        "       vec2 f = g.yx*vec2(10.0*mouse.x, 10.0*mouse.y);",
        // "        vec2 f = g.yx*vec2(-1.0,1.0);",

        "       g = mix( g, f, am1 );",

        "       uv -= 0.00005*g*am2;",
        "    }",

        "    vec3 col2 = texture2D(texture, uv).xyz;",
        "   vec2 q = vUv;",
        "   vec2 p = -1.0 + 2.0*q;",
        "   p.x *= resolution.x/resolution.y;",
        "   vec2 m = mouse;",
        "   m.x *= resolution.x/resolution.y;",
        "   float r = sqrt( dot((p - m), (p - m)) );",
        "   float a = atan(p.y, p.x);",
        "   vec3 col = texture2D(texture, vUv).rgb;",
        "   if(r < r2){",
        "       float f = smoothstep(r2, r2 - 0.5, r);",
        "       col = mix( col, col2, f);",
        "   }",
        "    gl_FragColor = vec4(col, 1.0);",
        "}"
    
    ].join("\n")

}
var ReposShader = function(){

    this.uniforms = THREE.UniformsUtils.merge( [

        {
            "texture"  : { type: "t", value: null },
            "mouse"  : { type: "v2", value: null },
            "resolution"  : { type: "v2", value: null },
            "time"  : { type: "f", value: null },
            "r2"  : { type: "f", value: null }

        }
    ] ),

    this.vertexShader = [

        "varying vec2 vUv;",

        "void main() {",
        "    vUv = uv;",
        "    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );",
        "}"
    
    ].join("\n"),
    
    this.fragmentShader = [
        

        "varying vec2 vUv;",
        "uniform sampler2D texture;",
        "uniform vec2 mouse;",
        "uniform vec2 resolution;",

        "uniform float r2;",

        "vec3 rgb2hsv(vec3 c)",
        "{",
        "    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);",
        "    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));",
        "    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));",
        "    ",
        "    float d = q.x - min(q.w, q.y);",
        "    float e = 1.0e-10;",
        "    return vec3(abs(( (q.z + (q.w - q.y) / (6.0 * d + e))) ), d / (q.x + e), q.x);",
        "}",

        "vec3 hsv2rgb(vec3 c)",
        "{",
        "    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);",
        "    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);",
        "    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);",
        "}",

        "void main(){",

        "    vec2 tc = vUv;",
        "    vec4 look = texture2D(texture,tc);",
        // "    vec2 offs = vec2(look.y-look.x,look.w-look.z)*0.001;",
        // "    vec2 offs = vec2(look.y-look.x,look.w-look.z)*vec2(mouse.x/3.333, mouse.y/3.333);",
        // "    vec2 offs = vec2(look.y-look.x,look.w-look.z)*vec2(mouse.x/50.0, mouse.y/50.0);",
        "    vec2 offs = vec2(look.y-look.x,look.w-look.z)*vec2(mouse.x/100.0, mouse.y/100.0);",
        // "    vec2 offs = vec2(look.y-look.x,look.w-look.z)*vec2(0.0, 0.01);",
        "    vec2 coord = offs+tc;",
        "    vec4 repos = texture2D(texture, coord);",
        // "    repos*=1.01;",
        // "    gl_FragColor = repos;",
        "  vec3 hsv = rgb2hsv(repos.rgb);",

        // "  hsv.r += 0.01;",
        // "  hsv.r = mod(hsv.r, 1.0);",
        // "  hsv.g *= 1.1;",
        // "  hsv.b *= 1.1;",
        // "  //hsv.g = mod(hsv.g, 1.0);",
        "  repos.rgb = hsv2rgb(hsv); ",
        // "    repos*=1.01;",

        "   vec2 q = vUv;",
        "   vec2 p = -1.0 + 2.0*q;",
        "   p.x *= resolution.x/resolution.y;",
        "   vec2 m = mouse;",
        "   m.x *= resolution.x/resolution.y;",
        "   float r = sqrt( dot((p - m), (p - m)) );",
        "   float a = atan(p.y, p.x);",
        "   vec3 col = texture2D(texture, vUv).rgb;",
        "   if(r < r2){",
        "       float f = smoothstep(r2, r2 - 0.5, r);",
        "       col = mix( col, repos.rgb, f);",
        "   }",
        "   gl_FragColor = vec4(col,1.0);",
        "}"
    
    ].join("\n")
    
}
var PaintShader = function(){

    this.uniforms = THREE.UniformsUtils.merge( [

        {
            "texture"  : { type: "t", value: null },
            "mouse"  : { type: "v2", value: null },
            "resolution"  : { type: "v2", value: null },
            "time"  : { type: "f", value: null }

        }
    ] ),

    this.vertexShader = [

        "varying vec2 vUv;",
        "void main() {",
        "    vUv = uv;",
        "    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );",
        "}"
    
    ].join("\n"),
    
    this.fragmentShader = [
        
        "uniform sampler2D texture; ",
        "uniform vec2 resolution; ",
        "varying vec2 vUv;",


         "const int radius = 1;",

         "void main() {",
        "    vec2 src_size = vec2 (1.0 / resolution.x, 1.0 / resolution.y);",
         "    //vec2 uv = gl_FragCoord.xy/resolution.xy;",
         "    vec2 uv = vUv;",
         "    float n = float((radius + 1) * (radius + 1));",
         "    int i; ",
        "    int j;",
         "    vec3 m0 = vec3(0.0); vec3 m1 = vec3(0.0); vec3 m2 = vec3(0.0); vec3 m3 = vec3(0.0);",
         "    vec3 s0 = vec3(0.0); vec3 s1 = vec3(0.0); vec3 s2 = vec3(0.0); vec3 s3 = vec3(0.0);",
         "    vec3 c;",

         "    for (int j = -radius; j <= 0; ++j)  {",
         "        for (int i = -radius; i <= 0; ++i)  {",
         "            c = texture2D(texture, uv + vec2(i,j) * src_size).rgb;",
         "            m0 += c;",
         "            s0 += c * c;",
         "        }",
         "    }",

         "    for (int j = -radius; j <= 0; ++j)  {",
         "        for (int i = 0; i <= radius; ++i)  {",
         "            c = texture2D(texture, uv + vec2(i,j) * src_size).rgb;",
         "            m1 += c;",
         "            s1 += c * c;",
         "        }",
         "    }",

         "    for (int j = 0; j <= radius; ++j)  {",
         "        for (int i = 0; i <= radius; ++i)  {",
         "            c = texture2D(texture, uv + vec2(i,j) * src_size).rgb;",
         "            m2 += c;",
         "            s2 += c * c;",
         "        }",
         "    }",

         "    for (int j = 0; j <= radius; ++j)  {",
         "        for (int i = -radius; i <= 0; ++i)  {",
         "            c = texture2D(texture, uv + vec2(i,j) * src_size).rgb;",
         "            m3 += c;",
         "            s3 += c * c;",
         "        }",
         "    }",


         "    float min_sigma2 = 1e+2;",
         "    m0 /= n;",
         "    s0 = abs(s0 / n - m0 * m0);",

         "    float sigma2 = s0.r + s0.g + s0.b;",
         "    if (sigma2 < min_sigma2) {",
         "        min_sigma2 = sigma2;",
         "        gl_FragColor = vec4(m0, 1.0);",
         "    }",

         "    m1 /= n;",
         "    s1 = abs(s1 / n - m1 * m1);",

         "    sigma2 = s1.r + s1.g + s1.b;",
         "    if (sigma2 < min_sigma2) {",
         "        min_sigma2 = sigma2;",
         "        gl_FragColor = vec4(m1, 1.0);",
         "    }",

         "    m2 /= n;",
         "    s2 = abs(s2 / n - m2 * m2);",

         "    sigma2 = s2.r + s2.g + s2.b;",
         "    if (sigma2 < min_sigma2) {",
         "        min_sigma2 = sigma2;",
         "        gl_FragColor = vec4(m2, 1.0);",
         "    }",

         "    m3 /= n;",
         "    s3 = abs(s3 / n - m3 * m3);",

         "    sigma2 = s3.r + s3.g + s3.b;",
         "    if (sigma2 < min_sigma2) {",
         "        min_sigma2 = sigma2;",
         "        gl_FragColor = vec4(m3, 1.0);",
         "    }",
         "}"
    
    ].join("\n")
    
}
var DenoiseShader = function(){
        this.uniforms = THREE.UniformsUtils.merge([
            {
                "texture"  : { type: "t", value: null },
                "mouse"  : { type: "v2", value: null },
                "resolution"  : { type: "v2", value: null },
                "time"  : { type: "f", value: null },
                "r2"  : { type: "f", value: null }

            }
        ]);

        this.vertexShader = [

            "varying vec2 vUv;",
            "void main() {",
            "    vUv = uv;",
            "    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );",
            "}"
        
        ].join("\n");
        
        this.fragmentShader = [
            
            "uniform sampler2D texture;",
            "uniform sampler2D alpha;",
            "uniform vec2 resolution;",
            "uniform vec2 mouse;",
            "uniform float r2;",
            "uniform float time;",
            "varying vec2 vUv;",

            "void main() {",

                "vec3 col = texture2D(texture, vUv).rgb;",

                "vec4 center = texture2D(texture, vUv);",
                "float exponent = 1.0;",
                "vec4 color = vec4(0.0);",
                "float total = 0.0;",
                "for (float x = -4.0; x <= 4.0; x += 1.0) {",
                "    for (float y = -4.0; y <= 4.0; y += 1.0) {",
                "        vec4 sample = texture2D(texture, vUv + vec2(x, y) / resolution);",
                "        float weight = 1.0 - abs(dot(sample.rgb - center.rgb, vec3(0.25)));",
                "        weight = pow(weight, exponent);",
                "        color += sample * weight;",
                "        total += weight;",
                "    }",
                "}",
                "vec4 col2 = color / total;",
                
                
                "gl_FragColor = vec4(col,1.0);",
            "}"


        
        ].join("\n");
}
var WarpShader = function(){
        this.uniforms = THREE.UniformsUtils.merge([
            {
                "texture"  : { type: "t", value: null },
                "texture2"  : { type: "t", value: null },
                "mouse"  : { type: "v2", value: null },
                "resolution"  : { type: "v2", value: null },
                "time"  : { type: "f", value: null },
                "r2"  : { type: "f", value: null }

            }
        ]);

        this.vertexShader = [

            "varying vec2 vUv;",
            "void main() {",
            "    vUv = uv;",
            "    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );",
            "}"
        
        ].join("\n");
        
        this.fragmentShader = [
            


            "uniform sampler2D texture;",
            "uniform sampler2D texture2;",

            "uniform sampler2D alpha;",
            "uniform vec2 resolution;",
            "uniform vec2 mouse;",
            "uniform float r2;",
            "uniform float time;",
            "varying vec2 vUv;",

            "void main(){",

            "    // vec2 uv = gl_FragCoord.xy / size.xy;",
            "    vec2 uv = vUv;",

            "vec3 col = texture2D(texture, vUv).rgb;",

            "vec4 center = texture2D(texture, vUv);",
            "float exponent = 1.0;",
            "vec4 color = vec4(0.0);",
            "float total = 0.0;",
            "for (float x = -4.0; x <= 4.0; x += 1.0) {",
            "    for (float y = -4.0; y <= 4.0; y += 1.0) {",
            "        vec4 sample = texture2D(texture2, vUv + vec2(x, y) / resolution);",
            "        float weight = 1.0 - abs(dot(sample.rgb - center.rgb, vec3(0.25)));",
            "        weight = pow(weight, exponent);",
            "        color += sample * weight;",
            "        total += weight;",
            "    }",
            "}",
            "vec4 col2 = color / total;",

            // "    vec4 video = texture2D(texture, uv);",
            "    vec4 video = col2;",
            "    float val3 = sin(time);",
            "    vec2 mouse2 = vec2(sin(time), cos(time))*0.1;",
            "    vec2 ray = vec2(uv.x - mouse2.x + mouse2.x * video.x + val3 * video.z,uv.y + mouse2.y - mouse2.y * video.y + val3 * video.z);",

            "    vec4 newVideo = texture2D(texture2, ray);",
            "    gl_FragColor = vec4(newVideo.xyz,1.0);",

            "}"


        
        ].join("\n");
}
var WarpShader2 = function(){
        this.uniforms = THREE.UniformsUtils.merge([
            {
                "texture"  : { type: "t", value: null },
                "mouse"  : { type: "v2", value: null },
                "resolution"  : { type: "v2", value: null },
                "time"  : { type: "f", value: null },
                "r2"  : { type: "f", value: null },
                "distortion"  : { type: "f", value: null },
                "distance"  : { type: "f", value: null },
                "speed"  : { type: "f", value: null }

            }
        ]);

        this.vertexShader = [

            "varying vec2 vUv;",
            "void main() {",
            "    vUv = uv;",
            "    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );",
            "}"
        
        ].join("\n");
        
        this.fragmentShader = [
            


            "uniform sampler2D texture;",

            "uniform sampler2D alpha;",
            "uniform vec2 resolution;",
            "uniform vec2 mouse;",
            "uniform float r2;",
            "uniform float time;",
            "uniform float speed;",
            "uniform float distortion;",
            "uniform float distance;",
            "varying vec2 vUv;",

            "void main(){",

            "    // vec2 uv = gl_FragCoord.xy / size.xy;",
            "    vec2 uv = vUv;",

            "vec3 col = texture2D(texture, vUv).rgb;",

            "vec4 center = texture2D(texture, vUv);",
            "float exponent = 1.0;",
            "vec4 color = vec4(0.0);",
            "float total = 0.0;",
            "for (float x = -4.0; x <= 4.0; x += 1.0) {",
            "    for (float y = -4.0; y <= 4.0; y += 1.0) {",
            "        vec4 sample = texture2D(texture, vUv + vec2(x, y) / resolution);",
            "        float weight = 1.0 - abs(dot(sample.rgb - center.rgb, vec3(0.25)));",
            "        weight = pow(weight, exponent);",
            "        color += sample * weight;",
            "        total += weight;",
            "    }",
            "}",
            "vec4 col2 = color / total;",

            // "    vec4 video = texture2D(texture, uv);",
            "    vec4 video = col2;",
            "    float osc = sin(time*speed)*distortion;",
            "    vec2 mouse2 = vec2(sin(time*speed), cos(time*speed))*distance;",
            "    vec2 ray = vec2(uv.x - mouse2.x + mouse2.x * video.x + osc * video.z,uv.y + mouse2.y - mouse2.y * video.y + osc * video.z);",

            "    vec4 newVideo = texture2D(texture, ray);",
            "    gl_FragColor = vec4(newVideo.xyz,1.0);",

            "}"


        
        ].join("\n");
}

var CurvesShader = function(red, green, blue){
        function clamp(lo, value, hi) {
            return Math.max(lo, Math.min(value, hi));
        }
        function splineInterpolate(points) {
            var interpolator = new SplineInterpolator(points);
            var array = [];
            for (var i = 0; i < 256; i++) {
                array.push(clamp(0, Math.floor(interpolator.interpolate(i / 255) * 256), 255));
            }
            return array;
        }

        red = splineInterpolate(red);
        if (arguments.length == 1) {
            green = blue = red;
        } else {
            green = splineInterpolate(green);
            blue = splineInterpolate(blue);
        }
        // createCanvas(red, green, blue);
        var array = [];
        for (var i = 0; i < 256; i++) {
            array.splice(array.length, 0, red[i], green[i], blue[i], 255);
        }
        curveMap = new THREE.DataTexture(array, 256, 1, THREE.RGBAFormat, THREE.UnsignedByteType);
        curveMap.minFilter = curveMap.magFilter = THREE.LinearFilter;
        curveMap.needsUpdate = true;
        // var noiseSize = 256;
        var size = 256;
        var data = new Uint8Array( 4 * size );
        for ( var i = 0; i < size * 4; i ++ ) {
            data[ i ] = array[i] | 0;
        }
        dt = new THREE.DataTexture( data, 256, 1, THREE.RGBAFormat );
        // dt.wrapS = THREE.ClampToEdgeWrapping;
        // dt.wrapT = THREE.ClampToEdgeWrapping;
        dt.needsUpdate = true;
        // console.log(dt);
        this.uniforms = THREE.UniformsUtils.merge([
            {
                "texture"  : { type: "t", value: null },
                "curveMap"  : { type: "t", value: dt },
                "mouse"  : { type: "v2", value: null },
                "resolution"  : { type: "v2", value: null },
                "time"  : { type: "f", value: null }

            }
        ]);

        this.vertexShader = [

            "varying vec2 vUv;",
            "void main() {",
            "    vUv = uv;",
            "    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );",
            "}"
        
        ].join("\n");
        
        this.fragmentShader = [
            
            "uniform sampler2D texture;",
            "uniform sampler2D curveMap;",
            "uniform vec2 resolution;",
            "uniform vec2 mouse;",
            "uniform float time;",
            "varying vec2 vUv;",

            "void main(){",
            
            "   vec4 curveColor = texture2D(texture, vUv);",
            "   curveColor.r = texture2D(curveMap, vec2(curveColor.r)).r;",
            "   curveColor.g = texture2D(curveMap, vec2(curveColor.g)).g;",
            "   curveColor.b = texture2D(curveMap, vec2(curveColor.b)).b;",

            "   gl_FragColor = vec4(curveColor.rgb,1.0);",
            "}",


        
        ].join("\n");
}

function SplineInterpolator(points) {
    var n = points.length;
    this.xa = [];
    this.ya = [];
    this.u = [];
    this.y2 = [];

    points.sort(function(a, b) {
        return a[0] - b[0];
    });
    for (var i = 0; i < n; i++) {
        this.xa.push(points[i][0]);
        this.ya.push(points[i][1]);
    }

    this.u[0] = 0;
    this.y2[0] = 0;

    for (var i = 1; i < n - 1; ++i) {
        // This is the decomposition loop of the tridiagonal algorithm. 
        // y2 and u are used for temporary storage of the decomposed factors.
        var wx = this.xa[i + 1] - this.xa[i - 1];
        var sig = (this.xa[i] - this.xa[i - 1]) / wx;
        var p = sig * this.y2[i - 1] + 2.0;

        this.y2[i] = (sig - 1.0) / p;

        var ddydx = 
            (this.ya[i + 1] - this.ya[i]) / (this.xa[i + 1] - this.xa[i]) - 
            (this.ya[i] - this.ya[i - 1]) / (this.xa[i] - this.xa[i - 1]);

        this.u[i] = (6.0 * ddydx / wx - sig * this.u[i - 1]) / p;
    }

    this.y2[n - 1] = 0;

    // This is the backsubstitution loop of the tridiagonal algorithm
    for (var i = n - 2; i >= 0; --i) {
        this.y2[i] = this.y2[i] * this.y2[i + 1] + this.u[i];
    }
}

SplineInterpolator.prototype.interpolate = function(x) {
    var n = this.ya.length;
    var klo = 0;
    var khi = n - 1;

    // We will find the right place in the table by means of
    // bisection. This is optimal if sequential calls to this
    // routine are at random values of x. If sequential calls
    // are in order, and closely spaced, one would do better
    // to store previous values of klo and khi.
    while (khi - klo > 1) {
        var k = (khi + klo) >> 1;

        if (this.xa[k] > x) {
            khi = k; 
        } else {
            klo = k;
        }
    }

    var h = this.xa[khi] - this.xa[klo];
    var a = (this.xa[khi] - x) / h;
    var b = (x - this.xa[klo]) / h;

    // Cubic spline polynomial is now evaluated.
    return a * this.ya[klo] + b * this.ya[khi] + 
        ((a * a * a - a) * this.y2[klo] + (b * b * b - b) * this.y2[khi]) * (h * h) / 6.0;
};