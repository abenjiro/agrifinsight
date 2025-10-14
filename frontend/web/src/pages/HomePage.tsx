import { useEffect } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, Camera, BarChart3, Lightbulb, Shield } from "lucide-react";

// amCharts imports
import * as am5 from "@amcharts/amcharts5";
import * as am5map from "@amcharts/amcharts5/map";
import am5geodata_worldLow from "@amcharts/amcharts5-geodata/worldLow";
import am5themes_Animated from "@amcharts/amcharts5/themes/Animated";

import homeImage from "../assets/img/home1.png";

export function HomePage() {
  useEffect(() => {
    // Initialize amCharts root
    const root = am5.Root.new("chartdiv");
    root.setThemes([am5themes_Animated.new(root)]);

    // Create map chart
    const chart = root.container.children.push(
      am5map.MapChart.new(root, {
        projection: am5map.geoOrthographic(),
        panX: "rotateX",
        panY: "rotateY",
        wheelX: "none",
        wheelY: "none",
      })
    );

    // Add world polygons
    const polygonSeries = chart.series.push(
      am5map.MapPolygonSeries.new(root, {
        geoJSON: am5geodata_worldLow,
      })
    );

    polygonSeries.mapPolygons.template.setAll({
      tooltipText: "{name}",
      interactive: true,
      fill: am5.color(0xcccccc),
    });

    // Highlight country
    const highlightCountry = (id: string) => {
      polygonSeries.mapPolygons.each((polygon) => {
        const countryId = polygon.dataItem?.dataContext ? (polygon.dataItem.dataContext as any).id : null;
        polygon.setAll({
          fill:
            countryId === id ? am5.color(0xff5733) : am5.color(0xcccccc),
        });
      });
    };

    // Animate globe rotation
    const selectCountry = (id: string) => {
      const dataItem = polygonSeries.getDataItemById(id);
      if (!dataItem) return;

      const polygon = dataItem.get("mapPolygon");
      if (polygon) {
        const centroid = polygon.geoCentroid();
        if (centroid) {
          chart.animate({
            key: "rotationX",
            to: -centroid.longitude,
            duration: 1500,
            easing: am5.ease.inOut(am5.ease.cubic),
          });
          chart.animate({
            key: "rotationY",
            to: -centroid.latitude,
            duration: 1500,
            easing: am5.ease.inOut(am5.ease.cubic),
          });
        }
        highlightCountry(id);
      }
    };

    // Agriculture facts
    const agricFacts = [
  { countryId: "GH", fact: "🇬🇭 Ghana produces about 20% of the world’s cocoa supply." },
  { countryId: "NG", fact: "🇳🇬 Nigeria is the world’s largest yam producer and among top cassava growers." },
  { countryId: "ET", fact: "🇪🇹 Ethiopia is Africa’s top coffee producer and the birthplace of Arabica coffee." },
  { countryId: "KE", fact: "🇰🇪 Kenya is famous for its high-quality tea, one of the country's top exports." },
  { countryId: "CI", fact: "🇨🇮 Côte d’Ivoire is the world’s largest cocoa producer, ahead of Ghana." },
  { countryId: "TZ", fact: "🇹🇿 Tanzania is a leading exporter of cashew nuts and cloves." },
  { countryId: "EG", fact: "🇪🇬 Egypt’s Nile Delta supports vast wheat and rice farming regions." },
  { countryId: "ZA", fact: "🇿🇦 South Africa leads in wine, citrus, and maize production in Africa." },
  { countryId: "UG", fact: "🇺🇬 Uganda is Africa’s largest producer of robusta coffee." },
  { countryId: "SD", fact: "🇸🇩 Sudan is one of the world’s major producers of gum arabic." },
  { countryId: "MW", fact: "🇲🇼 Malawi’s economy is heavily driven by tobacco and tea exports." },
  { countryId: "SN", fact: "🇸🇳 Senegal is a top groundnut (peanut) producer in West Africa." },
  { countryId: "BF", fact: "🇧🇫 Burkina Faso is a key cotton producer in sub-Saharan Africa." },
  { countryId: "BR", fact: "🇧🇷 Brazil is the world’s largest exporter of coffee and soybeans." },
  { countryId: "IN", fact: "🇮🇳 India is the world’s largest milk producer and second in rice production." },
];


    const factBox = document.getElementById("factbox");
    let index = 0;

    const showNextFact = () => {
      const { countryId, fact } = agricFacts[index];
      selectCountry(countryId);
      if (factBox) factBox.innerText = fact;
      index = (index + 1) % agricFacts.length;
    };

    showNextFact();
    const interval = setInterval(showNextFact, 5000);
    chart.appear(1000, 100);

    return () => {
      clearInterval(interval);
      root.dispose();
    };
  }, []);

  return (
    <div
      className="min-h-screen bg-cover bg-no-repeat"
      style={{ backgroundImage: `url(${homeImage})` }}
    >
      {/* Hero Section */}
      <section className="max-w-7xl mx-auto px-6 lg:px-8 pt-24 pb-20">
        <div className="flex flex-col lg:flex-row items-center justify-between gap-10">
          {/* Globe + Facts */}
          <div className="flex flex-col items-center lg:items-start w-full lg:w-1/2">
            <div
              id="chartdiv"
              className="w-full max-w-md h-[400px] rounded-full shadow-md"
            />
            <div
              id="factbox"
              className="mt-4 text-center text-lg font-semibold text-gray-800 min-h-[60px] flex items-center justify-center"
            ></div>
          </div>

          {/* Text Section */}
          <div className="text-center lg:text-left lg:w-1/2">
            <h1 className="text-4xl md:text-6xl font-extrabold text-gray-900 mb-6 leading-tight">
              Smart Agriculture with
              <span className="text-green-600"> AI-Powered </span>
              Insights
            </h1>
            <p className="text-lg text-gray-700 mb-8 max-w-lg mx-auto lg:mx-0">
              Transform your farming with intelligent disease detection, crop
              analysis, and personalized recommendations powered by machine
              learning.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
              <Link
                to="/register"
                className="bg-green-600 hover:bg-green-700 text-white px-8 py-3 text-lg rounded-lg inline-flex items-center justify-center transition"
              >
                Get Started
                <ArrowRight className="ml-2 w-5 h-5" />
              </Link>
              <Link
                to="/login"
                className="border border-green-600 text-green-700 hover:bg-green-50 px-8 py-3 text-lg rounded-lg inline-flex items-center justify-center transition"
              >
                Sign In
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="bg-white/90 py-20">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Everything you need for modern farming
            </h2>
            <p className="text-lg text-gray-600">
              Our platform combines cutting-edge AI with practical farming
              knowledge.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {[
              {
                icon: <Camera className="w-6 h-6 text-green-600" />,
                title: "Disease Detection",
                desc: "Upload plant images to instantly identify diseases and get treatment recommendations.",
              },
              {
                icon: <BarChart3 className="w-6 h-6 text-amber-600" />,
                title: "Crop Analysis",
                desc: "Get detailed insights about your crops' health, growth patterns, and yield predictions.",
              },
              {
                icon: <Lightbulb className="w-6 h-6 text-blue-600" />,
                title: "Smart Recommendations",
                desc: "Receive personalized advice on planting, fertilizing, and harvesting based on your data.",
              },
              {
                icon: <Shield className="w-6 h-6 text-purple-600" />,
                title: "Weather Integration",
                desc: "Stay ahead with weather forecasts and climate-based farming recommendations.",
              },
            ].map((feature, i) => (
              <div
                key={i}
                className="bg-white shadow-md rounded-2xl p-6 hover:shadow-lg transition text-center"
              >
                <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center mx-auto mb-4">
                  {feature.icon}
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  {feature.title}
                </h3>
                <p className="text-gray-600">{feature.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="bg-green-600 py-20">
        <div className="max-w-7xl mx-auto px-6 lg:px-8 text-center text-white">
          <h2 className="text-3xl font-bold mb-4">
            Ready to revolutionize your farming?
          </h2>
          <p className="text-lg text-green-100 mb-8">
            Join thousands of farmers already using AgriFinSight to increase
            their yields.
          </p>
          <Link
            to="/register"
            className="bg-white text-green-700 hover:bg-gray-100 px-8 py-3 text-lg rounded-lg inline-flex items-center transition"
          >
            Start Here
            <ArrowRight className="ml-2 w-5 h-5" />
          </Link>
        </div>
      </section>
    </div>
  );
}

