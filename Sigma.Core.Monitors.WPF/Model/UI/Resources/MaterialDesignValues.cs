/* 
MIT License

Copyright (c) 2016 Florian Cäsar, Michael Plainer

For full license see LICENSE in the root directory of this project. 
*/

using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text.RegularExpressions;
using System.Windows.Media;
using MaterialDesignColors;

// ReSharper disable BuiltInTypeReferenceStyle

namespace Sigma.Core.Monitors.WPF.Model.UI.Resources
{
	public enum MaterialColour
	{
		Amber,
		Blue,
		BlueGrey,
		Brown,
		Cyan,
		DeepOrange,
		DeepPurple,
		Green,
		Grey,
		Inidigo,
		LightBlue,
		LightGreen,
		Lime,
		Orange,
		Pink,
		Purple,
		Red,
		Teal,
		Yellow
	}

	public static class MaterialColourExtension
	{
		public static string ToFormattedString(this MaterialColour col)
		{
			return Regex.Replace(col.ToString(), "(\\B[A-Z])", " $1");
		}
	}

	public enum PrimaryColour
	{
		Primary50 = 50,
		Primary100 = 100,
		Primary200 = 200,
		Primary300 = 300,
		Primary400 = 400,
		Primary500 = 500,
		Primary600 = 600,
		Primary700 = 700,
		Primary800 = 800,
		Primary900 = 900
	}

	public enum AccentColour
	{
		Accent100 = 100,
		Accent200 = 200,
		Accent400 = 400,
		Accent700 = 700
	}

	public static class MaterialDesignValues
	{
		//TODO: documentation
		//TODO: validate that case change from Primar50 -> PRIMARY
		private const string PRIMARY50 = nameof(PRIMARY50);
		private const string PRIMARY100 = nameof(PRIMARY100);
		private const string PRIMARY200 = nameof(PRIMARY200);
		private const string PRIMARY300 = nameof(PRIMARY300);
		private const string PRIMARY400 = nameof(PRIMARY400);
		private const string PRIMARY500 = nameof(PRIMARY500);
		private const string PRIMARY600 = nameof(PRIMARY600);
		private const string PRIMARY700 = nameof(PRIMARY700);
		private const string PRIMARY800 = nameof(PRIMARY800);
		private const string PRIMARY900 = nameof(PRIMARY900);

		private const string ACCENT100 = nameof(ACCENT100);
		private const string ACCENT200 = nameof(ACCENT200);
		//private const string ACCENT300 = nameof(ACCENT300);
		private const string ACCENT400 = nameof(ACCENT400);
		//private const string ACCENT500 = nameof(ACCENT500);
		//private const string ACCENT600 = nameof(ACCENT600);
		private const string ACCENT700 = nameof(ACCENT700);

		public static readonly Swatch Amber = new Swatch("amber",
			new[]
			{
				new Hue(PRIMARY100, new Color {A = 255, R = 255, G = 236, B = 179}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY200, new Color {A = 255, R = 255, G = 224, B = 130}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY300, new Color {A = 255, R = 255, G = 213, B = 79}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY400, new Color {A = 255, R = 255, G = 202, B = 40}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY50, new Color {A = 255, R = 255, G = 248, B = 225}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY500, new Color {A = 255, R = 255, G = 193, B = 7}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY600, new Color {A = 255, R = 255, G = 179, B = 0}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY700, new Color {A = 255, R = 255, G = 160, B = 0}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY800, new Color {A = 255, R = 255, G = 143, B = 0}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY900, new Color {A = 255, R = 255, G = 111, B = 0}, new Color {A = 255, R = 0, G = 0, B = 0})
			},
			new[]
			{
				new Hue(ACCENT100, new Color {A = 255, R = 255, G = 229, B = 127}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT200, new Color {A = 255, R = 255, G = 215, B = 64}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT400, new Color {A = 255, R = 255, G = 196, B = 0}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT700, new Color {A = 255, R = 255, G = 171, B = 0}, new Color {A = 255, R = 0, G = 0, B = 0})
			});

		public static readonly Swatch Blue = new Swatch("blue",
			new[]
			{
				new Hue(PRIMARY100, new Color {A = 255, R = 187, G = 222, B = 251}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY200, new Color {A = 255, R = 144, G = 202, B = 249}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY300, new Color {A = 255, R = 100, G = 181, B = 246}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY400, new Color {A = 255, R = 66, G = 165, B = 245}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY50, new Color {A = 255, R = 227, G = 242, B = 253}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY500, new Color {A = 255, R = 33, G = 150, B = 243}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY600, new Color {A = 255, R = 30, G = 136, B = 229}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY700, new Color {A = 255, R = 25, G = 118, B = 210}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY800, new Color {A = 255, R = 21, G = 101, B = 192}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY900, new Color {A = 255, R = 13, G = 71, B = 161}, new Color {A = 255, R = 255, G = 255, B = 255})
			},
			new[]
			{
				new Hue(ACCENT100, new Color {A = 255, R = 130, G = 177, B = 255}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT200, new Color {A = 255, R = 68, G = 138, B = 255}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(ACCENT400, new Color {A = 255, R = 41, G = 121, B = 255}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(ACCENT700, new Color {A = 255, R = 41, G = 98, B = 255}, new Color {A = 255, R = 255, G = 255, B = 255})
			});

		public static readonly Swatch BlueGrey = new Swatch("bluegrey",
			new[]
			{
				new Hue(PRIMARY100, new Color {A = 255, R = 207, G = 216, B = 220}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY200, new Color {A = 255, R = 176, G = 190, B = 197}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY300, new Color {A = 255, R = 144, G = 164, B = 174}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY400, new Color {A = 255, R = 120, G = 144, B = 156}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY50, new Color {A = 255, R = 236, G = 239, B = 241}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY500, new Color {A = 255, R = 96, G = 125, B = 139}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY600, new Color {A = 255, R = 84, G = 110, B = 122}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY700, new Color {A = 255, R = 69, G = 90, B = 100}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY800, new Color {A = 255, R = 55, G = 71, B = 79}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY900, new Color {A = 255, R = 38, G = 50, B = 56}, new Color {A = 255, R = 255, G = 255, B = 255})
			},
			new Hue[] {});

		public static readonly Swatch Brown = new Swatch("brown",
			new[]
			{
				new Hue(PRIMARY100, new Color {A = 255, R = 215, G = 204, B = 200}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY200, new Color {A = 255, R = 188, G = 170, B = 164}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY300, new Color {A = 255, R = 161, G = 136, B = 127}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY400, new Color {A = 255, R = 141, G = 110, B = 99}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY50, new Color {A = 255, R = 239, G = 235, B = 233}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY500, new Color {A = 255, R = 121, G = 85, B = 72}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY600, new Color {A = 255, R = 109, G = 76, B = 65}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY700, new Color {A = 255, R = 93, G = 64, B = 55}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY800, new Color {A = 255, R = 78, G = 52, B = 46}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY900, new Color {A = 255, R = 62, G = 39, B = 35}, new Color {A = 255, R = 255, G = 255, B = 255})
			},
			new Hue[] {});

		public static readonly Swatch Cyan = new Swatch("cyan",
			new[]
			{
				new Hue(PRIMARY100, new Color {A = 255, R = 178, G = 235, B = 242}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY200, new Color {A = 255, R = 128, G = 222, B = 234}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY300, new Color {A = 255, R = 77, G = 208, B = 225}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY400, new Color {A = 255, R = 38, G = 198, B = 218}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY50, new Color {A = 255, R = 224, G = 247, B = 250}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY500, new Color {A = 255, R = 0, G = 188, B = 212}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY600, new Color {A = 255, R = 0, G = 172, B = 193}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY700, new Color {A = 255, R = 0, G = 151, B = 167}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY800, new Color {A = 255, R = 0, G = 131, B = 143}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY900, new Color {A = 255, R = 0, G = 96, B = 100}, new Color {A = 255, R = 255, G = 255, B = 255})
			},
			new[]
			{
				new Hue(ACCENT100, new Color {A = 255, R = 132, G = 255, B = 255}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT200, new Color {A = 255, R = 24, G = 255, B = 255}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT400, new Color {A = 255, R = 0, G = 229, B = 255}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT700, new Color {A = 255, R = 0, G = 184, B = 212}, new Color {A = 255, R = 0, G = 0, B = 0})
			});

		public static readonly Swatch DeepOrange = new Swatch("deeporange",
			new[]
			{
				new Hue(PRIMARY100, new Color {A = 255, R = 255, G = 204, B = 188}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY200, new Color {A = 255, R = 255, G = 171, B = 145}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY300, new Color {A = 255, R = 255, G = 138, B = 101}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY400, new Color {A = 255, R = 255, G = 112, B = 67}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY50, new Color {A = 255, R = 251, G = 233, B = 231}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY500, new Color {A = 255, R = 255, G = 87, B = 34}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY600, new Color {A = 255, R = 244, G = 81, B = 30}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY700, new Color {A = 255, R = 230, G = 74, B = 25}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY800, new Color {A = 255, R = 216, G = 67, B = 21}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY900, new Color {A = 255, R = 191, G = 54, B = 12}, new Color {A = 255, R = 255, G = 255, B = 255})
			},
			new[]
			{
				new Hue(ACCENT100, new Color {A = 255, R = 255, G = 158, B = 128}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT200, new Color {A = 255, R = 255, G = 110, B = 64}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT400, new Color {A = 255, R = 255, G = 61, B = 0}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(ACCENT700, new Color {A = 255, R = 221, G = 44, B = 0}, new Color {A = 255, R = 255, G = 255, B = 255})
			});

		public static readonly Swatch DeepPurple = new Swatch("deeppurple",
			new[]
			{
				new Hue(PRIMARY100, new Color {A = 255, R = 209, G = 196, B = 233}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY200, new Color {A = 255, R = 179, G = 157, B = 219}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY300, new Color {A = 255, R = 149, G = 117, B = 205}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY400, new Color {A = 255, R = 126, G = 87, B = 194}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY50, new Color {A = 255, R = 237, G = 231, B = 246}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY500, new Color {A = 255, R = 103, G = 58, B = 183}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY600, new Color {A = 255, R = 94, G = 53, B = 177}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY700, new Color {A = 255, R = 81, G = 45, B = 168}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY800, new Color {A = 255, R = 69, G = 39, B = 160}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY900, new Color {A = 255, R = 49, G = 27, B = 146}, new Color {A = 255, R = 255, G = 255, B = 255})
			},
			new[]
			{
				new Hue(ACCENT100, new Color {A = 255, R = 179, G = 136, B = 255}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT200, new Color {A = 255, R = 124, G = 77, B = 255}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(ACCENT400, new Color {A = 255, R = 101, G = 31, B = 255}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(ACCENT700, new Color {A = 255, R = 98, G = 0, B = 234}, new Color {A = 255, R = 255, G = 255, B = 255})
			});

		public static readonly Swatch Green = new Swatch("green",
			new[]
			{
				new Hue(PRIMARY100, new Color {A = 255, R = 200, G = 230, B = 201}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY200, new Color {A = 255, R = 165, G = 214, B = 167}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY300, new Color {A = 255, R = 129, G = 199, B = 132}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY400, new Color {A = 255, R = 102, G = 187, B = 106}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY50, new Color {A = 255, R = 232, G = 245, B = 233}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY500, new Color {A = 255, R = 76, G = 175, B = 80}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY600, new Color {A = 255, R = 67, G = 160, B = 71}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY700, new Color {A = 255, R = 56, G = 142, B = 60}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY800, new Color {A = 255, R = 46, G = 125, B = 50}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY900, new Color {A = 255, R = 27, G = 94, B = 32}, new Color {A = 255, R = 255, G = 255, B = 255})
			},
			new[]
			{
				new Hue(ACCENT100, new Color {A = 255, R = 185, G = 246, B = 202}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT200, new Color {A = 255, R = 105, G = 240, B = 174}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT400, new Color {A = 255, R = 0, G = 230, B = 118}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT700, new Color {A = 255, R = 0, G = 200, B = 83}, new Color {A = 255, R = 0, G = 0, B = 0})
			});

		public static readonly Swatch Grey = new Swatch("grey",
			new[]
			{
				new Hue(PRIMARY100, new Color {A = 255, R = 245, G = 245, B = 245}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY200, new Color {A = 255, R = 238, G = 238, B = 238}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY300, new Color {A = 255, R = 224, G = 224, B = 224}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY400, new Color {A = 255, R = 189, G = 189, B = 189}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY50, new Color {A = 255, R = 250, G = 250, B = 250}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY500, new Color {A = 255, R = 158, G = 158, B = 158}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY600, new Color {A = 255, R = 117, G = 117, B = 117}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY700, new Color {A = 255, R = 97, G = 97, B = 97}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY800, new Color {A = 255, R = 66, G = 66, B = 66}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY900, new Color {A = 255, R = 33, G = 33, B = 33}, new Color {A = 255, R = 255, G = 255, B = 255})
			},
			new Hue[] {});

		public static readonly Swatch Indigo = new Swatch("indigo",
			new[]
			{
				new Hue(PRIMARY100, new Color {A = 255, R = 197, G = 202, B = 233}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY200, new Color {A = 255, R = 159, G = 168, B = 218}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY300, new Color {A = 255, R = 121, G = 134, B = 203}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY400, new Color {A = 255, R = 92, G = 107, B = 192}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY50, new Color {A = 255, R = 232, G = 234, B = 246}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY500, new Color {A = 255, R = 63, G = 81, B = 181}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY600, new Color {A = 255, R = 57, G = 73, B = 171}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY700, new Color {A = 255, R = 48, G = 63, B = 159}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY800, new Color {A = 255, R = 40, G = 53, B = 147}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY900, new Color {A = 255, R = 26, G = 35, B = 126}, new Color {A = 255, R = 255, G = 255, B = 255})
			},
			new[]
			{
				new Hue(ACCENT100, new Color {A = 255, R = 140, G = 158, B = 255}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT200, new Color {A = 255, R = 83, G = 109, B = 254}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(ACCENT400, new Color {A = 255, R = 61, G = 90, B = 254}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(ACCENT700, new Color {A = 255, R = 48, G = 79, B = 254}, new Color {A = 255, R = 255, G = 255, B = 255})
			});

		public static readonly Swatch LightBlue = new Swatch("lightblue",
			new[]
			{
				new Hue(PRIMARY100, new Color {A = 255, R = 179, G = 229, B = 252}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY200, new Color {A = 255, R = 129, G = 212, B = 250}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY300, new Color {A = 255, R = 79, G = 195, B = 247}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY400, new Color {A = 255, R = 41, G = 182, B = 246}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY50, new Color {A = 255, R = 225, G = 245, B = 254}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY500, new Color {A = 255, R = 3, G = 169, B = 244}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY600, new Color {A = 255, R = 3, G = 155, B = 229}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY700, new Color {A = 255, R = 2, G = 136, B = 209}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY800, new Color {A = 255, R = 2, G = 119, B = 189}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY900, new Color {A = 255, R = 1, G = 87, B = 155}, new Color {A = 255, R = 255, G = 255, B = 255})
			},
			new[]
			{
				new Hue(ACCENT100, new Color {A = 255, R = 128, G = 216, B = 255}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT200, new Color {A = 255, R = 64, G = 196, B = 255}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT400, new Color {A = 255, R = 0, G = 176, B = 255}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT700, new Color {A = 255, R = 0, G = 145, B = 234}, new Color {A = 255, R = 255, G = 255, B = 255})
			});

		public static readonly Swatch LightGreen = new Swatch("lightgreen",
			new[]
			{
				new Hue(PRIMARY100, new Color {A = 255, R = 220, G = 237, B = 200}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY200, new Color {A = 255, R = 197, G = 225, B = 165}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY300, new Color {A = 255, R = 174, G = 213, B = 129}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY400, new Color {A = 255, R = 156, G = 204, B = 101}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY50, new Color {A = 255, R = 241, G = 248, B = 233}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY500, new Color {A = 255, R = 139, G = 195, B = 74}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY600, new Color {A = 255, R = 124, G = 179, B = 66}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY700, new Color {A = 255, R = 104, G = 159, B = 56}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY800, new Color {A = 255, R = 85, G = 139, B = 47}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY900, new Color {A = 255, R = 51, G = 105, B = 30}, new Color {A = 255, R = 255, G = 255, B = 255})
			},
			new[]
			{
				new Hue(ACCENT100, new Color {A = 255, R = 204, G = 255, B = 144}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT200, new Color {A = 255, R = 178, G = 255, B = 89}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT400, new Color {A = 255, R = 118, G = 255, B = 3}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT700, new Color {A = 255, R = 100, G = 221, B = 23}, new Color {A = 255, R = 0, G = 0, B = 0})
			});

		public static readonly Swatch Lime = new Swatch("lime",
			new[]
			{
				new Hue(PRIMARY100, new Color {A = 255, R = 240, G = 244, B = 195}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY200, new Color {A = 255, R = 230, G = 238, B = 156}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY300, new Color {A = 255, R = 220, G = 231, B = 117}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY400, new Color {A = 255, R = 212, G = 225, B = 87}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY50, new Color {A = 255, R = 249, G = 251, B = 231}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY500, new Color {A = 255, R = 205, G = 220, B = 57}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY600, new Color {A = 255, R = 192, G = 202, B = 51}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY700, new Color {A = 255, R = 175, G = 180, B = 43}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY800, new Color {A = 255, R = 158, G = 157, B = 36}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY900, new Color {A = 255, R = 130, G = 119, B = 23}, new Color {A = 255, R = 255, G = 255, B = 255})
			},
			new[]
			{
				new Hue(ACCENT100, new Color {A = 255, R = 244, G = 255, B = 129}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT200, new Color {A = 255, R = 238, G = 255, B = 65}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT400, new Color {A = 255, R = 198, G = 255, B = 0}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT700, new Color {A = 255, R = 174, G = 234, B = 0}, new Color {A = 255, R = 0, G = 0, B = 0})
			});

		public static readonly Swatch Orange = new Swatch("orange",
			new[]
			{
				new Hue(PRIMARY100, new Color {A = 255, R = 255, G = 224, B = 178}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY200, new Color {A = 255, R = 255, G = 204, B = 128}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY300, new Color {A = 255, R = 255, G = 183, B = 77}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY400, new Color {A = 255, R = 255, G = 167, B = 38}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY50, new Color {A = 255, R = 255, G = 243, B = 224}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY500, new Color {A = 255, R = 255, G = 152, B = 0}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY600, new Color {A = 255, R = 251, G = 140, B = 0}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY700, new Color {A = 255, R = 245, G = 124, B = 0}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY800, new Color {A = 255, R = 239, G = 108, B = 0}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY900, new Color {A = 255, R = 230, G = 81, B = 0}, new Color {A = 255, R = 255, G = 255, B = 255})
			},
			new[]
			{
				new Hue(ACCENT100, new Color {A = 255, R = 255, G = 209, B = 128}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT200, new Color {A = 255, R = 255, G = 171, B = 64}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT400, new Color {A = 255, R = 255, G = 145, B = 0}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT700, new Color {A = 255, R = 255, G = 109, B = 0}, new Color {A = 255, R = 0, G = 0, B = 0})
			});

		public static readonly Swatch Pink = new Swatch("pink",
			new[]
			{
				new Hue(PRIMARY100, new Color {A = 255, R = 248, G = 187, B = 208}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY200, new Color {A = 255, R = 244, G = 143, B = 177}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY300, new Color {A = 255, R = 240, G = 98, B = 146}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY400, new Color {A = 255, R = 236, G = 64, B = 122}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY50, new Color {A = 255, R = 252, G = 228, B = 236}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY500, new Color {A = 255, R = 233, G = 30, B = 99}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY600, new Color {A = 255, R = 216, G = 27, B = 96}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY700, new Color {A = 255, R = 194, G = 24, B = 91}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY800, new Color {A = 255, R = 173, G = 20, B = 87}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY900, new Color {A = 255, R = 136, G = 14, B = 79}, new Color {A = 255, R = 255, G = 255, B = 255})
			},
			new[]
			{
				new Hue(ACCENT100, new Color {A = 255, R = 255, G = 128, B = 171}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT200, new Color {A = 255, R = 255, G = 64, B = 129}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(ACCENT400, new Color {A = 255, R = 245, G = 0, B = 87}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(ACCENT700, new Color {A = 255, R = 197, G = 17, B = 98}, new Color {A = 255, R = 255, G = 255, B = 255})
			});

		public static readonly Swatch Purple = new Swatch("purple",
			new[]
			{
				new Hue(PRIMARY100, new Color {A = 255, R = 225, G = 190, B = 231}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY200, new Color {A = 255, R = 206, G = 147, B = 216}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY300, new Color {A = 255, R = 186, G = 104, B = 200}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY400, new Color {A = 255, R = 171, G = 71, B = 188}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY50, new Color {A = 255, R = 243, G = 229, B = 245}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY500, new Color {A = 255, R = 156, G = 39, B = 176}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY600, new Color {A = 255, R = 142, G = 36, B = 170}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY700, new Color {A = 255, R = 123, G = 31, B = 162}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY800, new Color {A = 255, R = 106, G = 27, B = 154}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY900, new Color {A = 255, R = 74, G = 20, B = 140}, new Color {A = 255, R = 255, G = 255, B = 255})
			},
			new[]
			{
				new Hue(ACCENT100, new Color {A = 255, R = 234, G = 128, B = 252}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT200, new Color {A = 255, R = 224, G = 64, B = 251}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(ACCENT400, new Color {A = 255, R = 213, G = 0, B = 249}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(ACCENT700, new Color {A = 255, R = 170, G = 0, B = 255}, new Color {A = 255, R = 255, G = 255, B = 255})
			});

		public static readonly Swatch Red = new Swatch("red",
			new[]
			{
				new Hue(PRIMARY100, new Color {A = 255, R = 255, G = 205, B = 210}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY200, new Color {A = 255, R = 239, G = 154, B = 154}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY300, new Color {A = 255, R = 229, G = 115, B = 115}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY400, new Color {A = 255, R = 239, G = 83, B = 80}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY50, new Color {A = 255, R = 255, G = 235, B = 238}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY500, new Color {A = 255, R = 244, G = 67, B = 54}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY600, new Color {A = 255, R = 229, G = 57, B = 53}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY700, new Color {A = 255, R = 211, G = 47, B = 47}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY800, new Color {A = 255, R = 198, G = 40, B = 40}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY900, new Color {A = 255, R = 183, G = 28, B = 28}, new Color {A = 255, R = 255, G = 255, B = 255})
			},
			new[]
			{
				new Hue(ACCENT100, new Color {A = 255, R = 255, G = 138, B = 128}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT200, new Color {A = 255, R = 255, G = 82, B = 82}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(ACCENT400, new Color {A = 255, R = 255, G = 23, B = 68}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(ACCENT700, new Color {A = 255, R = 213, G = 0, B = 0}, new Color {A = 255, R = 255, G = 255, B = 255})
			});

		public static readonly Swatch Teal = new Swatch("teal",
			new[]
			{
				new Hue(PRIMARY100, new Color {A = 255, R = 178, G = 223, B = 219}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY200, new Color {A = 255, R = 128, G = 203, B = 196}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY300, new Color {A = 255, R = 77, G = 182, B = 172}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY400, new Color {A = 255, R = 38, G = 166, B = 154}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY50, new Color {A = 255, R = 224, G = 242, B = 241}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY500, new Color {A = 255, R = 0, G = 150, B = 136}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY600, new Color {A = 255, R = 0, G = 137, B = 123}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY700, new Color {A = 255, R = 0, G = 121, B = 107}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY800, new Color {A = 255, R = 0, G = 105, B = 92}, new Color {A = 255, R = 255, G = 255, B = 255}),
				new Hue(PRIMARY900, new Color {A = 255, R = 0, G = 77, B = 64}, new Color {A = 255, R = 255, G = 255, B = 255})
			},
			new[]
			{
				new Hue(ACCENT100, new Color {A = 255, R = 167, G = 255, B = 235}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT200, new Color {A = 255, R = 100, G = 255, B = 218}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT400, new Color {A = 255, R = 29, G = 233, B = 182}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT700, new Color {A = 255, R = 0, G = 191, B = 165}, new Color {A = 255, R = 0, G = 0, B = 0})
			});

		public static readonly Swatch Yellow = new Swatch("yellow",
			new[]
			{
				new Hue(PRIMARY100, new Color {A = 255, R = 255, G = 249, B = 196}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY200, new Color {A = 255, R = 255, G = 245, B = 157}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY300, new Color {A = 255, R = 255, G = 241, B = 118}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY400, new Color {A = 255, R = 255, G = 238, B = 88}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY50, new Color {A = 255, R = 255, G = 253, B = 231}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY500, new Color {A = 255, R = 255, G = 235, B = 59}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY600, new Color {A = 255, R = 253, G = 216, B = 53}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY700, new Color {A = 255, R = 251, G = 192, B = 45}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY800, new Color {A = 255, R = 249, G = 168, B = 37}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(PRIMARY900, new Color {A = 255, R = 245, G = 127, B = 23}, new Color {A = 255, R = 0, G = 0, B = 0})
			},
			new[]
			{
				new Hue(ACCENT100, new Color {A = 255, R = 255, G = 255, B = 141}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT200, new Color {A = 255, R = 255, G = 255, B = 0}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT400, new Color {A = 255, R = 255, G = 234, B = 0}, new Color {A = 255, R = 0, G = 0, B = 0}),
				new Hue(ACCENT700, new Color {A = 255, R = 255, G = 214, B = 0}, new Color {A = 255, R = 0, G = 0, B = 0})
			});


		private static Hue GetHue(IEnumerable<Hue> hues, string name)
		{
			foreach (Hue hue in hues)
			{
				if (string.Equals(hue.Name, name, StringComparison.CurrentCultureIgnoreCase))
				{
					return hue;
				}
			}

			throw new KeyNotFoundException(name);
		}

		public static Swatch GetSwatch(MaterialColour colour)
		{
			FieldInfo field = typeof(MaterialDesignValues).GetField(colour.ToString(), BindingFlags.Public | BindingFlags.Static);

			return field?.GetValue(null) as Swatch;
		}

		public static Color GetColour(MaterialColour colour, PrimaryColour primaryColour)
		{
			return GetColour(GetSwatch(colour), primaryColour);
		}

		public static Color GetColour(Swatch swatch, PrimaryColour primaryColour)
		{
			return GetHue(swatch.PrimaryHues, primaryColour.ToString()).Color;
		}

		public static Color GetForegroundColor(MaterialColour colour, PrimaryColour primaryColour)
		{
			return GetForegroundColor(GetSwatch(colour), primaryColour);
		}

		public static Color GetForegroundColor(Swatch swatch, PrimaryColour primaryColour)
		{
			return GetHue(swatch.PrimaryHues, primaryColour.ToString()).Foreground;
		}

		public static Color GetColour(MaterialColour colour, AccentColour accentColour)
		{
			return GetColour(GetSwatch(colour), accentColour);
		}

		public static Color GetColour(Swatch swatch, AccentColour accentColour)
		{
			return GetHue(swatch.AccentHues, accentColour.ToString()).Color;
		}

		public static Color GetForegroundColor(MaterialColour colour, AccentColour accentColour)
		{
			return GetForegroundColor(GetSwatch(colour), accentColour);
		}

		public static Color GetForegroundColor(Swatch swatch, AccentColour accentColour)
		{
			return GetHue(swatch.AccentHues, accentColour.ToString()).Foreground;
		}
	}
}