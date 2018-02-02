from PIL import Image
import sys

def main():	
	im = Image.open(sys.argv[1])
	width, height = im.size
	rgb = im.getpixel((1,1))

	for y in range(height):
		for x in range(width):
			rgb = im.getpixel((x, y))
			rgb = (rgb[0]//2, rgb[1]//2, rgb[2]//2)
			im.putpixel((x, y), rgb)

	im.save("Q2.png")

if __name__ == '__main__':
	main()