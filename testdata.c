#include <stdio.h>
#include <stdint.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "weights.h"
#include "biases.h"
#define MNIST_TEST_IMAGES_FILE "t10k-images-idx3-ubyte"
#define MNIST_TEST_LABELS_FILE "t10k-labels-idx1-ubyte"
#define MNIST_IMAGE_SIZE 784
#define NUM_IMAGES 10000
#define SHIFT 8
int Conv_3D_layer1(float *input, const float *weights, int64_t *out, const float * biases, 
            int kerenl_ch, int kernel_row, int kernel_col,int kernel_cnt, int ouput_row, int output_col, int input_row, int input_col)
{
  int i, j, k, l, s, p, n,m; 

  for(m = 0; m < kernel_cnt; m ++)            // 커널의 개수 변수 m
  {
    for(i = 0; i < ouput_row; i++)           //  ouput  행의 변수 i , 최상단 반복문
    {
      for(j = 0; j < output_col; j++)         // output 열의 변수 j
      {
        *(out + ouput_row * output_col * m  + output_col * i + j) = (int64_t)(*(biases + m)*SHIFT*SHIFT); //bias 연산
        for(n = 0; n < kerenl_ch; n ++)
          {
            l = 0;                     // kernel 행의 값 0 초기화
            for(k = i; k < i + kernel_row; k++) // input 행의 변수 k, output 행의 값에 따라 input Conv 시작 행 바뀜.
            {
                p = 0;                    //kernel 열의 값 0 초기화  
              for(s = j; s < j + kernel_col ; s++) // input 열의 변수 s, output 열의 값에 따라 input Conv 시작 열 바뀜.
              {
                *(out + ouput_row * output_col* m + output_col * i + j) += (int64_t)(*(input + n *input_row * input_col + k * input_col + s)*SHIFT) *                            
                                                                          (int64_t)(*(weights + kernel_col * kernel_row * kerenl_ch * m + kernel_col * kernel_row * n  + kernel_col * l + p)*SHIFT); 
                  p++;
              }                 // kernel 열의 값 1씩 증가 

            l++;                        // kernel 행의 값 1씩 증가
           }
        }
      }
    }   
  }
}

int Conv_3D_layer3(int64_t *input, const float *weights, int64_t *out, const float * biases, 
            int kerenl_ch, int kernel_row, int kernel_col,int kernel_cnt, int ouput_row, int output_col, int input_row, int input_col)
{
  int i, j, k, l, s, p, n,m; 

  for(m = 0; m < kernel_cnt; m ++)            // 커널의 개수 변수 m
  {
    for(i = 0; i < ouput_row; i++)           //  ouput  행의 변수 i , 최상단 반복문
    {
      for(j = 0; j < output_col; j++)         // output 열의 변수 j
      {
        *(out + ouput_row * output_col * m  + output_col * i + j) = (int64_t)(*(biases + m)*SHIFT*SHIFT*SHIFT); //bias 연산
        for(n = 0; n < kerenl_ch; n ++)
          {
            l = 0;                     // kernel 행의 값 0 초기화
            for(k = i; k < i + kernel_row; k++) // input 행의 변수 k, output 행의 값에 따라 input Conv 시작 행 바뀜.
            {
                p = 0;                    //kernel 열의 값 0 초기화  
              for(s = j; s < j + kernel_col ; s++) // input 열의 변수 s, output 열의 값에 따라 input Conv 시작 열 바뀜.
              {
                *(out + ouput_row * output_col* m + output_col * i + j) += *(input + n *input_row * input_col + k * input_col + s) *                            
                                                                          (int64_t)(*(weights + kernel_col * kernel_row * kerenl_ch * m + kernel_col * kernel_row * n  + kernel_col * l + p)*SHIFT); 
                  p++;
              }                 // kernel 열의 값 1씩 증가 

            l++;                        // kernel 행의 값 1씩 증가
           }
        }
      }
    }   
  }
}

// Activaction Function
  int Relu(int64_t *out, int out_cnt, int out_row, int out_col )
  {
    int i;
    for(i = 0; i < out_cnt * out_row * out_col; i ++) // 전체 SIZE
    {
      if( *(out + i) < 0)        // 0보다 작으면 0의 값 저장
      {
        *(out + i) = 0;
      }
      else
      {
        *(out + i) = *(out + i);  // 0보다 크거나 같으면 원래 값 저장
      }
    }
  }

  //  Out 초기화
  int Out_init(int64_t *out, int out_row, int out_col, int out_ch)
  {
    int i;
    for(i = 0; i < out_row * out_col * out_ch; i ++)
      {
        *(out+ i) = 0;
      }

  }    

 // MAX Pooling
  int Max_Pooling(int64_t *input, int64_t *out, int input_row, int input_col, int input_ch,int output_row, int output_col)
  {
    int i, j, l, k, s;
    float max_1,max_2, max;
    for( s = 0; s < input_ch; s++)
    {
      k = 0;
      for(i = 0; i < input_row; i = i + 2)
      {
        l = 0;
        for(j = 0; j < input_col; j = j + 2)
        {
            max_1 = (*(input + input_row * input_col * s + input_col * i + j) > *(input + input_row * input_col * s + input_col * i + j + 1)) ?
             *(input + input_row * input_col * s + input_col * i + j) : *(input + input_row * input_col * s +input_col * i + j + 1) ;                 // 2x2 filter 윗행 비교 연산

            max_2 = (*(input + input_row * input_col * s + input_col * (i + 1) + j) > *(input + input_row * input_col * s + input_col * (i + 1) + j + 1)) ?
             *(input + input_row * input_col * s +input_col * (i + 1) + j) : *(input + input_row * input_col * s + input_col * (i + 1) + j + 1) ;      // 2x2 filter 아래행 비교 연산

            max = (max_1 > max_2) ? max_1 : max_2 ;         //  2x2 filter Max 값 저장

            *(out + output_row * output_col * s +  k * output_col + l) = max ;  // output에 Max 값 저장
            l++;
            

        }  
        k++;
      }    
    }
  }
  int Average_Pooling(float *input, float *out, int input_row, int input_col, int input_ch,int output_row, int output_col)
  {
    int i, j ,s, l, k;
    float avg; 
    for(s = 0; s < input_ch; s++)
    {
      k = 0;
      for(i = 0; i < input_row; i = i + 2)
      {
        l = 0;
        for(j = 0; j < input_col; j = j + 2)
        {
          avg = (*(input + input_row * input_col * s + input_col * i + j) + *(input + input_row * input_col * s  + input_col * i + j + 1) +
                *(input + input_row * input_col * s + input_col * (i+ 1) + j) + *(input + input_row * input_col * s  + input_col * (i + 1) + j + 1)) / 4.0;

          *(out + output_row * output_col * s +  k * output_col + l) = avg ;
          l++;
        }
        k ++;
      }
    }
  }

int Full_Connection_layer5(int64_t *input, int64_t *out, const float * weights, const float *biases, int out_size, int input_size)
{
  int i, j;
  for(i = 0; i < out_size; i++)                     // output size
  {
    *(out + i) =  (int64_t)(*(biases + i)*SHIFT*SHIFT*SHIFT*SHIFT);           // bias 연산
    for(j = 0; j < input_size; j++)                // input size
    {
      *(out + i) += *(input + j) * (int64_t)(*(weights + input_size * i + j)*SHIFT);   // full_connection 연산
    }
  }
}
int Full_Connection_layer6(int64_t *input, int64_t *out, const float * weights, const float *biases, int out_size, int input_size)
{
  int i, j;
  for(i = 0; i < out_size; i++)                     // output size
  {
    *(out + i) =  (int64_t)(*(biases + i)*SHIFT*SHIFT*SHIFT*SHIFT*SHIFT);          // bias 연산
    for(j = 0; j < input_size; j++)                // input size
    {
      *(out + i) += *(input + j) * (int64_t)(*(weights + input_size * i + j)*SHIFT);    // full_connection 연산
    }
  }
}
int Full_Connection_layer7(int64_t *input, int64_t *out, const float * weights, const float *biases, int out_size, int input_size)
{
  int i, j;
  for(i = 0; i < out_size; i++)                     // output size
  {
    *(out + i) =  (int64_t)(*(biases + i)*SHIFT*SHIFT*SHIFT*SHIFT*SHIFT*SHIFT);          // bias 연산
    for(j = 0; j < input_size; j++)                // input size
    {
      *(out + i) += *(input + j) * (int64_t)(*(weights + input_size * i + j)*SHIFT);   // full_connection 연산
    }
  }
}
// Activation Function (1D)
int Relu_1D(int64_t *out, int out_size)
{
  int i;
  for(i = 0; i <  out_size; i ++ )
  {
    if (*(out + i) < 0)
    {
      *(out + i) = 0;
    }
    else
    {
      *(out + i) = *(out + i);
    }
  }
}
// PADDING
int Padding(uint8_t *Input,float *Out, int output_row, int output_col, int input_row, int input_col, int padding_size )
{
  int i, j;
    for(i = 0; i < output_row ; i++)
    {
       for(j = 0; j < output_col; j++)
       {
            if((i < padding_size)||(i > input_row + padding_size - 1)||(j < padding_size)||(j > input_col + padding_size - 1))      
            {
                *(Out + i*output_col+ j) = 0;
            }
            else
            {
                *(Out + i*output_col + j) = *(Input + (i - padding_size)*input_col + j - padding_size)/255.0;
            }
       } 
    }
}
int findMax(float *input,int maxlable) {
    float max = *input;
    maxlable = 0;
    for (int sp = 1; sp < 9; sp++) {
        if (*(input + sp) > max) {
            max = *(input + sp);
            maxlable = sp;
        }
    }
}



int main() {
    FILE *images_file = fopen(MNIST_TEST_IMAGES_FILE, "rb");
    FILE *labels_file = fopen(MNIST_TEST_LABELS_FILE, "rb");

    if (!images_file || !labels_file) {
        printf("Failed to open MNIST test files.\n");
        return 1;
    }

    // Read the header information (discard for now)
    uint32_t magic_number, num_images, num_rows, num_cols;
    fread(&magic_number, sizeof(uint32_t), 1, images_file);
    fread(&num_images, sizeof(uint32_t), 1, images_file);
    fread(&num_rows, sizeof(uint32_t), 1, images_file);
    fread(&num_cols, sizeof(uint32_t), 1, images_file);

    magic_number = __builtin_bswap32(magic_number);
    num_images = __builtin_bswap32(num_images);
    num_rows = __builtin_bswap32(num_rows);
    num_cols = __builtin_bswap32(num_cols);

    // Read the labels header
    uint32_t labels_magic_number, num_labels;
    fread(&labels_magic_number, sizeof(uint32_t), 1, labels_file);
    fread(&num_labels, sizeof(uint32_t), 1, labels_file);

    labels_magic_number = __builtin_bswap32(labels_magic_number);
    num_labels = __builtin_bswap32(num_labels);



    //fread(image, sizeof(uint8_t), num_rows * num_cols, images_file);
    uint8_t label[NUM_IMAGES];
    fread(&label, sizeof(uint8_t), NUM_IMAGES, labels_file);

    uint8_t **images = (uint8_t **)malloc(sizeof(uint8_t *) * NUM_IMAGES);
    for (int i = 0; i < NUM_IMAGES; i++) {
        images[i] = (uint8_t *)malloc(sizeof(uint8_t) * MNIST_IMAGE_SIZE);
    }

    // Read and store each image
    for (int i = 0; i < NUM_IMAGES; i++) {
        fread(images[i], sizeof(uint8_t), MNIST_IMAGE_SIZE, images_file);


    }
    
    int accuracy;
    
    for(int label_size = 0; label_size < NUM_IMAGES; label_size++)
    {
     
  /*pImg = stbi_load("3.png", &Width, &height, &channels, 0);*/
  
  //Padding (input = 28 x 28 x 1 output = 32x32x1, PAD = 2)
  float *Pad_input = (float*)malloc(sizeof(float)*32*32);
  Padding(images[label_size], Pad_input, 32, 32, 28, 28, 2);
  

  //layer 1 (3D Conv, input = 32 x 32 x 1, output = 28 x 28 x 6, kernel = 5 x 5 x 1 x 6)
  const float *w1 = &weights_C1[0][0][0][0];
  const float *b1 = &biases_C1[0];
  int64_t *out_1 = (int64_t*)malloc(sizeof(int64_t)*28*28*6);
  Conv_3D_layer1(Pad_input, w1, out_1, b1, 1, 5, 5, 6, 28, 28, 32, 32);
  Relu(out_1, 6, 28, 28);
 


  // layer 2 (Max Poolong, input = 28 x 28 x 6, output = 14 x 14 x 6)
  int64_t *out_2 = (int64_t*)malloc(sizeof(int64_t)*14*14*6);
  Max_Pooling(out_1, out_2, 28,28,6,14,14);
  
    
  //layer 3 (3D Conv, input = 14 x 14 x 6, output = 10 x 10 x 16, kernel = 5 x 5 x 6 x 16)
  const float *w2 = &weights_C2[0][0][0][0];
  const float *b2 = &biases_C2[0];
  int64_t *out_3 = (int64_t*)malloc(sizeof(int64_t)*10*10*16);
  Conv_3D_layer3(out_2, w2, out_3, b2, 6, 5, 5, 16, 10, 10, 14, 14);
  Relu(out_3, 16, 10, 10);

  //layer 4 (Max Pooling, input = 10 x 10 x 16, output = 5 x 5 x 16)
  int64_t *out_4 = (int64_t*)malloc(sizeof(int64_t)*5*5*16);
  Max_Pooling(out_3, out_4, 10,10,16,5,5);

 //layer 5 (FC, input = 400, output = 120)
  const float *w3 = &weights_F1[0][0];
  const float *b3 = &biases_F1[0];
  int64_t *out_5 = (int64_t*)malloc(sizeof(int64_t)*120);
  Full_Connection_layer5(out_4, out_5, w3, b3, 120, 400 );
  Relu_1D(out_5, 120);
  
 //layer 6 (FC, input = 120, output = 84)
  const float *w4 = &weights_F2[0][0];
  const float *b4 = &biases_F2[0];
  int64_t *out_6 = (int64_t*)malloc(sizeof(int64_t)*84);
  Full_Connection_layer6(out_5, out_6, w4, b4, 84, 120 );
  Relu_1D(out_6, 84);

  //layer 7 (FC, input = 84, output = 10)
  const float *w5 = &weights_F3[0][0];
  const float *b5 = &biases_F3[0];
  int64_t *out_7 = (int64_t*)malloc(sizeof(int64_t)*10);
  Full_Connection_layer7(out_6, out_7, w5, b5, 10, 84 );

  //숫자값 추론
  float max = (*out_7)/SHIFT/SHIFT/SHIFT/SHIFT/SHIFT/SHIFT;
  int maxlable = 0;
/*
   for(int r = 0; r < 10; r++)
  {
    printf("%d : %f\n",r, (float)*(out_7 + r)/256/256/256/256/256/256);
  }
*/
   for(int k = 1; k < 10; k++)
  {
    if(max < *(out_7+ k)/SHIFT/SHIFT/SHIFT/SHIFT/SHIFT/SHIFT)
    {
      max = *(out_7+ k)/SHIFT/SHIFT/SHIFT/SHIFT/SHIFT/SHIFT;
      maxlable = k;
    }
  }
  if(maxlable == label[label_size])
    {
      accuracy = accuracy + 1;
    }
  //printf("%d : %f\n", maxlable, max);
  free(out_1);
  free(out_2);
  free(out_3);
  free(out_4);
  free(out_5);
  free(out_6);
  free(out_7);
  free(images[label_size]);
    
  }

  printf("%s : %d", "Accuracy",accuracy);    
    fclose(images_file);
    fclose(labels_file);
}


    fclose(labels_file);
}

