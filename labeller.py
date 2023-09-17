from paddleocr import PaddleOCR
import os
import csv

ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=True, show_log = False)

with open('.\\data.csv', 'w') as csv_file:
    writer = csv.DictWriter(csv_file, extrasaction='ignore', fieldnames=['file', 'desc'])
    gcount = 0
    for subdir, dir, files in os.walk("train_imgs"):
        try:
            count = 0
            for file in files:
                if file.endswith('.png') or file.endswith('.jpg'):
                    print(f'File {count}')
                    location = f'.\\{os.path.join(subdir, file)}'

                    results = ocr.ocr(f'{location}', cls=False, det=True, rec=True)
                    print(f'Result: {results}')
                    
                    
                    text = ''
                    if results == [[]]:
                        if subdir != 'train_imgs':
                            text = subdir[11:]           
                            print(f'Text: {text}') 
                            os.rename(location, f'.\\processed_imgs\\{count}.png')                  
                            writer.writerow({'file': f'{count}.png', 'desc': text})

                    else:
                        words = []
                        for result in results[0]:
                            if 'mematic' not in result[1][0] and '.com' not in result[1][0] and 'kapwing' not in result[1][0]:
                                words.append(result[1][0])
                        text = ' '.join(words)               
                        print(f'Text: {text}')               
                        os.rename(location, f'.\\processed_imgs\\{count}.png')       
                        writer.writerow({'file': f'{count}.png', 'desc': text})
                                        
                    count += 1
                    gcount += 1
        except:
            pass

print(f'Total memes read: {gcount}')
        