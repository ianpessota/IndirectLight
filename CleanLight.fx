//Shader for contrast/birght


uniform bool debug = false;


uniform float BloomLight<
	ui_type = "slider";
ui_min = 0; ui_max = 2.0; ui_step = 0.05;
> = 0;
uniform float BloomLevel <
	ui_type = "slider";
ui_min = 0; ui_max = 2.0; ui_step = 0.05;
> = 0.25;


uniform float Blend<
	ui_type = "slider";
ui_min = 0; ui_max = 1.0; ui_step = 0.05;
> = 0;
uniform bool IndirectLight = false;

uniform float reflect<
	ui_type = "slider";
ui_min = 0; ui_max =0.1; ui_step = 0.005;
> = 0;
uniform float ShadowLevel<
	ui_type = "slider";
ui_min = 0; ui_max = 3.0; ui_step = 0.05;
> = 0.25;




static const float3 lumaCoeff = float3(0.2126f,0.7152f,0.0722f);
#define ZNEAR 0.3
#define ZFAR 80.0
#include "ReShade.fxh"


#include "ReShade.fxh"

texture DepthBTex : DEPTH;
sampler sDepthBTex { Texture = DepthBTex; };

texture colorTex  { Width = BUFFER_WIDTH*0.5; Height = BUFFER_HEIGHT*0.5; Format = RGBA16F; };
sampler colorSamp { Texture = colorTex; };

float4 PS_color(float4 vpos : SV_Position, float2 texcoord : TEXCOORD): SV_Target
{
	float4 color = tex2D(ReShade::BackBuffer, texcoord)*2.12;
	return saturate(float4(color.rgb,1));
}

float linearDepth(float2 txCoords)
{
	return (2.0*ZNEAR)/(ZFAR+ZNEAR-tex2D(sDepthBTex,txCoords).x*(ZFAR-ZNEAR));
}

texture AATex  { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA8; };
sampler AASamp { Texture = AATex; };

texture AATex0  { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA8; };
sampler AASamp0 { Texture = AATex0; };


float3 getColor(float2 texcoord)
{
	float3 color=tex2D(colorSamp, texcoord).rgb;

	color *= tex2D(colorSamp, float2(texcoord.x, 1/BUFFER_WIDTH*ReShade::PixelSize.y)).rgb;
	color *= tex2D(colorSamp, float2(texcoord.x, 1/BUFFER_WIDTH*ReShade::PixelSize.y)).rgb;
	color *= tex2D(colorSamp, float2(1/BUFFER_HEIGHT*ReShade::PixelSize.x, texcoord.y)).rgb;
	color *= tex2D(colorSamp, float2(1/BUFFER_HEIGHT*ReShade::PixelSize.x, texcoord.y)).rgb;

	color*=0.25;
	color += float4(0.004,0.39,0.55,0).rgb;
	return lerp(color,tex2D(colorSamp,  texcoord.y).rgb,0.2)*1.2;
}
float3 HighLight(float3 base)
{
	float adapt = 1;

	float max = dot(base, float3(1,1,1))*0.33;
	float min = 1 - max;
	if (max < min)
		max = min;
	adapt = lerp(0.0, 1.0, smoothstep(0.0, 1.0, max));
	adapt = lerp(min,max, step(min, adapt));

	float3 color = base*adapt;						  
	
	color+=base*dot(color, float3(adapt, adapt, adapt));
	return dot(color, float3(1,1,1));
}
float3 LightSpeed(float3 base, float3 colorInput)
{
	float c  = 0.33*dot(base, colorInput);
	
	float3 blank = HighLight(base);
	float3 black = dot(base, float3(0.22, 0.707, 0.071));

	base = lerp(base, blank, smoothstep(0.0, 1.0, step(0.250, c)));
	base = lerp(base, black, smoothstep(0.0, 1.0, step(0.250, c)));

	base +=base*c;

	return dot(base,float3(0.22, 0.707, 0.071));
}
float3 PS_ShadowLight(float3 base, float2 texcoord, float depth, float3 SLight)
{
	
	float3 ori = base;
	
	
	float3 Color =SLight*base+base;
	float3 sunColor = lerp(ori,float3(base.r*Color.r,base.g*Color.g,base.b*Color.b)*1.2,0.5);
	float3 Shadow = Color;


	depth *= 0.035;
	
	//Shadow
	base = Shadow;
	Shadow =lerp(ori, base, 0.5);
	Shadow = lerp(Shadow,float3(0,0,0),0.5);
	
	//Color
	base=ori;
	base = Color*pow(2,depth+1);

	Color=lerp(ori,sunColor,0.5);

	return saturate(lerp(dot(Color,Shadow), dot(Shadow,float3(0.22, 0.71, 0.07)),5));
}
float3 MidColor(float3 base, float2 texcoord,float i)
{
	float3 c=base;

			c.r+= tex2D(ReShade::BackBuffer,texcoord + float2(0.0, ReShade::PixelSize.y)).r;
			c.r+= tex2D(ReShade::BackBuffer,texcoord - float2(0.0, ReShade::PixelSize.y)).r ;
			c.r+= tex2D(ReShade::BackBuffer,texcoord + float2( ReShade::PixelSize.x,0.0)).r;
			c.r+= tex2D(ReShade::BackBuffer,texcoord - float2( ReShade::PixelSize.x,0.0)).r;
			c.r *= 0.2;
			
			c.g+= tex2D(ReShade::BackBuffer,texcoord + float2( 0,ReShade::PixelSize.y)).g;
			c.g+= tex2D(ReShade::BackBuffer,texcoord - float2(0, ReShade::PixelSize.y)).g ;
			c.g+= tex2D(ReShade::BackBuffer,texcoord + float2( ReShade::PixelSize.x,0.0)).g;
			c.g+= tex2D(ReShade::BackBuffer,texcoord - float2(ReShade::PixelSize.x,0.0)).g;
			c.g *= 0.2;

			c.b+= tex2D(ReShade::BackBuffer,texcoord + float2( 0,ReShade::PixelSize.y)).b;
			c.b+= tex2D(ReShade::BackBuffer,texcoord - float2( 0,ReShade::PixelSize.y)).b ;
			c.b+= tex2D(ReShade::BackBuffer,texcoord + float2(ReShade::PixelSize.x,0.0)).b;
			c.b+= tex2D(ReShade::BackBuffer,texcoord - float2( ReShade::PixelSize.x,0.0)).b;
			c.b *= 0.2;
			return lerp(base,c,i);
}
float colorComp(float3 c1, float3 c2, float depth)
{
	float a  = c1.r+c1.g+c1.b;
	float b  = c2.r+c2.g+c2.b;
	return a/b;
}
float4 PS_AA0(float4 vpos : SV_Position, float2 texcoord : TEXCOORD ) : SV_Target
{
	float4 Color;
	float depth =linearDepth(texcoord) ;

	
				const float2 offset[8] = {
		float2(1.0, 1.0),
		float2(0.0, -1.0),
		float2(-1.0, 1.0),
		float2(-1.0, -1.0),
		float2(0.0, 1.0),
		float2(0.0, -1.0),
		float2(1.0, 0.0),
		float2(-1.0, 0.0)
	};

			float4 c=tex2D(ReShade::BackBuffer, texcoord);

			c.r+= tex2D(ReShade::BackBuffer,texcoord + float2(0.0, ReShade::PixelSize.y)).r;
			c.r+= tex2D(ReShade::BackBuffer,texcoord - float2(0.0, ReShade::PixelSize.y)).r ;
			c.r+= tex2D(ReShade::BackBuffer,texcoord + float2( ReShade::PixelSize.x,0.0)).r;
			c.r+= tex2D(ReShade::BackBuffer,texcoord - float2( ReShade::PixelSize.x,0.0)).r;
			c.r *= 0.2;
			
			c.g+= tex2D(ReShade::BackBuffer,texcoord + float2( 0,ReShade::PixelSize.y)).g;
			c.g+= tex2D(ReShade::BackBuffer,texcoord - float2(0, ReShade::PixelSize.y)).g ;
			c.g+= tex2D(ReShade::BackBuffer,texcoord + float2( ReShade::PixelSize.x,0.0)).g;
			c.g+= tex2D(ReShade::BackBuffer,texcoord - float2(ReShade::PixelSize.x,0.0)).g;
			c.g *= 0.2;

			c.b+= tex2D(ReShade::BackBuffer,texcoord + float2( 0,ReShade::PixelSize.y)).b;
			c.b+= tex2D(ReShade::BackBuffer,texcoord - float2( 0,ReShade::PixelSize.y)).b ;
			c.b+= tex2D(ReShade::BackBuffer,texcoord + float2(ReShade::PixelSize.x,0.0)).b;
			c.b+= tex2D(ReShade::BackBuffer,texcoord - float2( ReShade::PixelSize.x,0.0)).b;
			c.b *= 0.2;
			
	for (int i = 0; i < 4; i++)
	{
		float2 bloomuv = offset[i] * ReShade::PixelSize.xy * 2;
		bloomuv += texcoord;
		float4 tempbloom = tex2Dlod(ReShade::BackBuffer, float4(bloomuv.xy, 0, 0));
		tempbloom.w = max(0, dot(tempbloom.xyz, 0.333) );
		tempbloom.xyz = max(0, tempbloom.xyz - 0.85); 
		c+= tempbloom;
	}
			Color=float4(saturate(c.rgb-0.25),c.a);
			return lerp(Color,(float4(getColor(texcoord)*Color,1)+c)*(pow(depth,4)/4),0.25);
	
}
float4 PS_AA(float4 vpos : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
	float4 Color;
	float depth =linearDepth(texcoord) ;

	
			float4 n,s,w,e;
			
			n= tex2D(ReShade::BackBuffer, float2(texcoord.x, texcoord.y+1/BUFFER_HEIGHT*4));
			s= tex2D(ReShade::BackBuffer, float2(texcoord.x, texcoord.y-1/BUFFER_HEIGHT*4));
			w= tex2D(ReShade::BackBuffer, float2(texcoord.x-1/BUFFER_WIDTH*4, texcoord.y));
			e= tex2D(ReShade::BackBuffer, float2(texcoord.x+1/BUFFER_WIDTH*4, texcoord.y));
			Color= (n+s+w+e+tex2D(ReShade::BackBuffer, texcoord))*0.2;
			
		
	
const float2 offset[8] = {
		float2(0.707, 0.707),
		float2(0.707, -0.707),
		float2(-0.707, 0.707),
		float2(-0.707, -0.707),
		float2(0.0, 1.0),
		float2(0.0, -1.0),
		float2(1.0, 0.0),
		float2(-1.0, 0.0)
	};

			float4 c=tex2D(AASamp0, texcoord);

			c.r+= tex2D(AASamp0,texcoord + float2(0.0, ReShade::PixelSize.y)).r;
			c.r+= tex2D(AASamp0,texcoord - float2(0.0, ReShade::PixelSize.y)).r ;
			c.r+= tex2D(AASamp0,texcoord + float2( ReShade::PixelSize.x,0.0)).r;
			c.r+= tex2D(AASamp0,texcoord - float2( ReShade::PixelSize.x,0.0)).r;
			c.r *= 0.2;
			
			c.g+= tex2D(AASamp0,texcoord + float2( 0,ReShade::PixelSize.y)).g;
			c.g+= tex2D(AASamp0,texcoord - float2(0, ReShade::PixelSize.y)).g ;
			c.g+= tex2D(AASamp0,texcoord + float2( ReShade::PixelSize.x,0.0)).g;
			c.g+= tex2D(AASamp0,texcoord - float2(ReShade::PixelSize.x,0.0)).g;
			c.g *= 0.2;

			c.b+= tex2D(AASamp0,texcoord + float2( 0,ReShade::PixelSize.y)).b;
			c.b+= tex2D(AASamp0,texcoord - float2( 0,ReShade::PixelSize.y)).b ;
			c.b+= tex2D(AASamp0,texcoord + float2(ReShade::PixelSize.x,0.0)).b;
			c.b+= tex2D(AASamp0,texcoord - float2( ReShade::PixelSize.x,0.0)).b;
			c.b*=0.2;


			for (int i = 0; i < 8; i++)
			{
				float2 bloomuv = offset[i] * ReShade::PixelSize * 8;
				bloomuv += texcoord;
				c+= tex2Dlod(AASamp0, float4(bloomuv, 0, 0));
			}

			Color=lerp(Color, lerp(Color,c,0.65), clamp(0,1,depth))*BloomLight;
			return saturate(Color);
	
}

float4 ReflectFX(float4 base, float2 texcoord) 
{

	float4 color= tex2D(ReShade::BackBuffer,texcoord);
	if(!IndirectLight)
		return  tex2D(ReShade::BackBuffer,texcoord);
	
	float ldepth =linearDepth(texcoord) ;
	

	float ndepth =linearDepth(float2(texcoord.x,texcoord.y+0.005));

	float2 perspectiveCorrection = float2(2.0f * (0.5 - texcoord.x) * texcoord.y, 2.0f * (0.5 - texcoord.y) * texcoord.x);

	//float p =  pow(2,(ldepth))-1;
	//float c = tex2D(ReShade::BackBuffer,texcoord).r+tex2D(ReShade::BackBuffer,texcoord).b+tex2D(ReShade::BackBuffer,texcoord).g;
	//c=(c*0.33);

	float blend = 1;// max(0.05, (1 - p));

	
	float angle = smoothstep(0,3.14,acos(ndepth/ldepth))*100;

	

	//c=abs(c-(color.r+color.g+color.b));


	
	color.a=ldepth*angle;

	float4 r = tex2D(ReShade::BackBuffer,float2(texcoord.x+0.05*perspectiveCorrection.x,clamp(0,1,1-texcoord.y)));
	if(ldepth<texcoord.y)
		blend = clamp(0,1,smoothstep(0,0.25,ldepth)*clamp(0,1,1-color.a) * colorComp(tex2D(ReShade::BackBuffer,texcoord).rgb,r.rgb,0));
	
	
	return lerp(tex2D(ReShade::BackBuffer,texcoord), r, blend);


	
}

float4 GaussBlur22(float2 coord, sampler tex, float mult, float lodlevel, bool isBlurVert)
{
	float4 sum = 0;
	float2 axis = isBlurVert ? float2(0, 1) : float2(1, 0);

	const float weight[11] = {
		0.082607,
		0.080977,
		0.076276,
		0.069041,
		0.060049,
		0.050187,
		0.040306,
		0.031105,
		0.023066,
		0.016436,
		0.011254
	};

	for (int i = -10; i < 11; i++)
	{
		float currweight = weight[abs(i)];
		sum += tex2Dlod(tex, float4(coord.xy + axis.xy * (float)i * ReShade::PixelSize * mult, 0, lodlevel)) * currweight;
	}

	return sum;
}
void PS_Light(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 outColor : SV_Target)  //base 0.36 perf ave
{
			///////// Variables ///////////////////////////////////////////////////////////////////

			float3 base = tex2D(ReShade::BackBuffer, texcoord).rgb;
			
			float3 originalBase = base;
			float3 bloom =1;
			float3 otherColor= float3(0,0,0);
			float3 lightColor = base;
			
			float3 ColorRGB = float3(0.35,2.25,0.75);
			float3 rgb_ =1;
		
		
				rgb_ = base*base;
				rgb_.r*=dot(base,float3(0.25,0.95,0.05));
				rgb_.g*=dot(base,float3(0.685,0.75,0.6));
				rgb_.b*=dot(base,float3(0.05,0.5,0.96));
			float depth =linearDepth(texcoord) ;
			float ldepth = depth;
			

			float c = tex2D(ReShade::BackBuffer,texcoord).r+tex2D(ReShade::BackBuffer,texcoord).b+tex2D(ReShade::BackBuffer,texcoord).g;
	c=(c*0.33);
	

		
			ldepth=clamp(0.1,1,sin(ldepth/3.14));
			float ndepth =linearDepth(float2(texcoord.x,texcoord.y+0.005));
			float angle = smoothstep(0,3.14,acos(ndepth/depth))*100;
			
			float edepth =linearDepth(float2(texcoord.x+angle *ReShade::PixelSize.x,texcoord.y))*0.99;
			float wdepth =linearDepth(float2(texcoord.x-angle *ReShade::PixelSize.x,texcoord.y)) * 0.99;
			float angleE = clamp(0,6.28,acos(edepth/depth));
			float angleW = clamp(0,6.28,acos(wdepth/depth));
			float angleN = clamp(0,6.28,acos(ndepth/depth));

			
				base=lerp(base, tex2D(AASamp, texcoord).rgb,clamp(0.0,0.5,0.75*ldepth)).rgb;
			
				originalBase = base;
			
			
			
			///////// Global Ilumination ///////////////////
			float3 preGI = originalBase;
			const float2 offset[4] = {
		float2(1.0, 1.0),
		float2(1.0, 1.0),
		float2(-1.0, 1.0),
		float2(-1.0, -1.0)
	};
				[unroll]
				for (int i = 0; i <4; i++)
				{
					preGI *= LightSpeed(base, preGI);
					preGI *= (lerp(base, originalBase, 0.5) - depth*(i*0.05));
					preGI *= i*preGI-0.01;
					preGI -= tex2D(ReShade::BackBuffer, texcoord + float2(0.5, -0.5)*offset[i]*ReShade::PixelSize).rgb;
					preGI += tex2D(ReShade::BackBuffer, texcoord + float2(1.5, -1.5)*offset[i]* ReShade::PixelSize).rgb;
					preGI += tex2D(ReShade::BackBuffer, texcoord + float2(0.5, 0.5)*offset[i]* ReShade::PixelSize).rgb;
					preGI -= tex2D(ReShade::BackBuffer, texcoord + float2(1.5, 1.5)*offset[i]*ReShade::PixelSize).rgb;
					preGI = saturate(preGI);
					
				}
				
				
			
			/////////  Brightness (Light)   /////////////////////////////////////////////////////////////////////////

			float3 Color = base;
			otherColor = base;
			float3 basecolor;
			float3 basedark;
			base = LightSpeed(originalBase, 0.1);
			float3 SLight=base;
		
			
			Color = lerp(Color, saturate(Color*clamp(base, 0.05, 1)*1.25), 0.25)*2;
			otherColor=lerp(Color, Color*clamp(base, 0, 1)*1.25, 0.5)*2;

			//saturation
			float3 hl = dot(pow(base, 2), float3(0.5, 0.5, 1));
			base = hl*float3(0.5, 0.5, 1) * 2 - 1;
			
			//variance
			hl = normalize(base);
			base = 0.5*saturate(dot(hl, Color*Color));

			base = base*HighLight(base);
			
			basedark=MidColor(base,texcoord,0.7)*LightSpeed(base,1);
					
			float3 finalColor = lerp(base, Color, smoothstep(0, 1,5));
			
			finalColor  = lerp(base, saturate(finalColor),0.65)*max(SLight,2.5);
			base = saturate(lerp(finalColor,base,0.5));
		

			
			///////// GI Second "Pass" ////////////////////////////////////////

			
				float3 GI = originalBase;
				
				int i = 0;
			
				GI += 0.5*LightSpeed(originalBase, preGI);
				GI = lerp(GI, originalBase, 0.55)*GI;
				[unroll]
				for (i = 0; i <4; i++)
				{
					GI += LightSpeed(originalBase, 0.1*i);
				}
				preGI = lerp(saturate(preGI+GI), originalBase, 0.25)*1.5;
		
				preGI = saturate(lerp(preGI,MidColor(preGI, texcoord,0.75),0.75));
			
				base=saturate(lerp(saturate(base*preGI*2), originalBase,0.85));
				base+= 0.15*dot(base*0.5,originalBase);
				base-= 0.35*PS_ShadowLight(base, texcoord, depth, SLight);
				

			
		
			////////   Bright and Saturation   ////////////////////////////////////	////////////////
			float dnts =1;
		
				dnts-=angleE;
				dnts-=angleW;
				dnts=1-dnts;
				dnts*=depth*angle;
				if(dnts>0.85)
				dnts=0;
				else if(dnts>0.5)
				dnts=0.85;
				else
				dnts=1;
			if(angle>6.28 || angle <0.05)
			dnts=0;
			finalColor*=dnts*(1-depth);
			base=lerp(base,base-(1-basedark*rgb_)*0.15,ShadowLevel*clamp(0,1,min(ldepth,0.75)));
			Color=MidColor(base, texcoord,0.95 ) ;

			
			
			base= lerp(base+0.175*SLight, Color ,(pow(2,ldepth)-1));
			
		
			float4 base4;
			float4 reflectTex =1;
			for (int ji = 0; ji <6; ji++)
				{
					
			
					
					Color = 0.01*tex2D(ReShade::BackBuffer, float2(texcoord.x+c/BUFFER_WIDTH, texcoord.y+c/BUFFER_HEIGHT)).rgb;
					Color  += 0.01*tex2D(ReShade::BackBuffer, float2(texcoord.x-c/BUFFER_WIDTH, texcoord.y+c/BUFFER_HEIGHT)).rgb;
					Color  += 0.01*tex2D(ReShade::BackBuffer, float2(texcoord.x+c/BUFFER_WIDTH, texcoord.y-c/BUFFER_HEIGHT)).rgb;
					Color  += 0.01*tex2D(ReShade::BackBuffer, float2(texcoord.x-c/BUFFER_WIDTH, texcoord.y-c/BUFFER_HEIGHT)).rgb;
					
				
					base+=Color*(clamp(0,1,log10(depth))*0.1);

					
				}
					
				
			if(IndirectLight)
			{
				base4 = GaussBlur22(texcoord.xy, AASamp, 16, 0, 0);
				base4.w *= 0.95;
				base4.xyz *= BloomLevel * base4.w;
				base4 = float4(saturate(max(base4.rgb, Color) - 0.125 * getColor(texcoord)), 1);
				base = lerp(base, base4.rgb, 0.2);
				reflectTex = ReflectFX(float4(base,1),texcoord);
				reflectTex = float4(base+clamp(0,1,finalColor)*reflectTex.rgb*clamp(0,1,texcoord.y*3),1);
				float v=clamp(0.1,1,abs((angle-.25)*3.14/8+depth))*dot(dnts*clamp(0,1,1-depth),reflectTex)*max((1-reflectTex.a),c);
		
			
				base4 = lerp(clamp(0.35,1,angle*0.25)*float4(originalBase,1),
						
						float4(base,1-reflectTex.a) 
					,1)+clamp(0,reflect*(1-depth),v);

				
					
			}
			else
			{
				base4 = GaussBlur22(texcoord.xy, AASamp, 16, 0, 0);
				base4.w *= 0.95;
				base4.xyz *= 0 * base4.w;
				base4 = float4(saturate(max(base4.rgb, Color) - 0.125 * getColor(texcoord)), 1);
				base = lerp(base, base4.rgb, 0.2);
			}
			
	
			
			
		if(debug)
		{ 
			if(angle>=1) 
				angle=1-depth; 
			outColor = float4(base+(finalColor)*reflectTex.rgb*clamp(0,1,texcoord.y*3)*(1-depth),1);//float4(clamp(0,1,angle),0,0,0);
		}
		else
			outColor = float4(lerp(base4.rgb,originalBase,clamp(0,1,0.85-ldepth)),1);
			
		base=lerp(base,outColor.rgb,1-ldepth).rgb;
		outColor=lerp(float4(base+(finalColor)*reflectTex.rgb*clamp(0,1,texcoord.y*3)*(depth),1),outColor, Blend);
}



technique LightFilter
{
		
		
		
		
		pass ColorPass
		{
			VertexShader = PostProcessVS;
			PixelShader = PS_color;
			RenderTarget = colorTex;
		}
		pass PreAAPass
		{
			VertexShader = PostProcessVS;
			PixelShader = PS_AA0;
			RenderTarget = AATex0;
		}
		pass AAPass
		{
			VertexShader = PostProcessVS;
			PixelShader = PS_AA;
			RenderTarget = AATex;
		}
		pass LightPass
		{
			VertexShader = PostProcessVS;
			PixelShader = PS_Light;	
			ClearRenderTargets = true;
		}


	
}
