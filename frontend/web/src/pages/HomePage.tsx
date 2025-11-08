import { useEffect } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, Camera, BarChart3, Lightbulb, Shield, Sparkles, TrendingUp, Zap, CheckCircle2, Star } from "lucide-react";

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
      className="min-h-screen bg-cover bg-no-repeat relative"
      style={{ backgroundImage: `url(${homeImage})` }}
    >
      {/* Animated Background Overlay */}
      <div className="fixed inset-0 -z-10 pointer-events-none">
        <div className="absolute top-20 left-10 w-72 h-72 bg-green-400/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute top-40 right-20 w-96 h-96 bg-blue-400/15 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
        <div className="absolute bottom-20 left-1/3 w-80 h-80 bg-yellow-400/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
      </div>

      {/* Hero Section */}
      <section className="max-w-7xl mx-auto px-6 lg:px-8 pt-24 pb-20">
        <div className="flex flex-col lg:flex-row items-center justify-between gap-10">
          {/* Globe + Facts */}
          <div className="flex flex-col items-center lg:items-start w-full lg:w-1/2">
            <div className="relative w-full max-w-md">
              <div className="absolute -inset-4 bg-gradient-to-br from-green-400 to-blue-500 rounded-full blur-2xl opacity-20 animate-pulse"></div>
              <div className="relative bg-white/10 backdrop-blur-sm rounded-3xl p-4 shadow-2xl border-2 border-white/30">
                <div
                  id="chartdiv"
                  className="w-full h-[400px]"
                  style={{ borderRadius: '24px', overflow: 'hidden' }}
                />
              </div>
            </div>
            <div
              id="factbox"
              className="mt-6 text-center text-lg font-semibold text-gray-800 min-h-[60px] flex items-center justify-center bg-white/80 backdrop-blur-md rounded-2xl px-6 py-4 shadow-lg border border-green-200 w-full max-w-md"
            ></div>
          </div>

          {/* Text Section */}
          <div className="text-center lg:text-left lg:w-1/2">
            {/* <div className="inline-flex items-center gap-2 px-4 py-2 mb-6 bg-gradient-to-r from-green-500 to-blue-500 rounded-full shadow-lg animate-pulse">
              <Sparkles className="w-5 h-5 text-white" />
              <span className="text-sm font-bold text-white">AI-Powered Innovation</span>
            </div> */}

            <h1 className="text-4xl md:text-6xl font-extrabold text-gray-900 mb-6 leading-tight">
              Smart Agriculture with
              <span className="bg-gradient-to-r from-green-600 via-emerald-600 to-blue-600 bg-clip-text text-transparent"> AI-Powered </span>
              Insights
            </h1>
            <p className="text-xl text-gray-700 mb-6 max-w-lg mx-auto lg:mx-0 leading-relaxed">
              Transform your farming with intelligent disease detection, crop
              analysis, and personalized recommendations powered by machine
              learning.
            </p>

            {/* Stats Pills */}
            <div className="flex flex-wrap gap-3 justify-center lg:justify-start mb-8">
              <div className="inline-flex items-center gap-2 px-4 py-2 bg-white/80 backdrop-blur-sm rounded-full shadow-md border border-green-200">
                <TrendingUp className="w-4 h-4 text-green-600" />
                <span className="text-sm font-semibold text-gray-800">35% Yield Increase</span>
              </div>
              <div className="inline-flex items-center gap-2 px-4 py-2 bg-white/80 backdrop-blur-sm rounded-full shadow-md border border-blue-200">
                <Zap className="w-4 h-4 text-blue-600" />
                <span className="text-sm font-semibold text-gray-800">98% Accuracy</span>
              </div>
              <div className="inline-flex items-center gap-2 px-4 py-2 bg-white/80 backdrop-blur-sm rounded-full shadow-md border border-purple-200">
                <Star className="w-4 h-4 text-purple-600" />
                <span className="text-sm font-semibold text-gray-800">50K+ Farmers</span>
              </div>
            </div>

            <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
              <Link
                to="/register"
                className="group bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white px-8 py-4 text-lg rounded-xl inline-flex items-center justify-center transition-all shadow-lg hover:shadow-xl hover:scale-105"
              >
                Get Started
                <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </Link>
              <Link
                to="/ai-features"
                className="border-2 border-green-600 text-green-700 hover:bg-green-50 px-8 py-4 text-lg rounded-xl inline-flex items-center justify-center transition-all shadow-md hover:shadow-lg hover:scale-105"
              >
                Explore AI Features
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="bg-gradient-to-b from-white/90 to-green-50/80 py-20">
        <div className="max-w-7xl mx-auto px-6 lg:px-8">
          <div className="text-center mb-16">
            <div className="inline-block mb-4">
              <div className="flex items-center gap-2 px-5 py-2 bg-gradient-to-r from-green-100 to-blue-100 rounded-full border-2 border-green-200">
                <Sparkles className="w-5 h-5 text-green-600" />
                <span className="text-sm font-bold text-green-700">Powerful Features</span>
              </div>
            </div>
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
              Everything you need for
              <span className="bg-gradient-to-r from-green-600 to-blue-600 bg-clip-text text-transparent"> modern farming</span>
            </h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Our platform combines cutting-edge AI with practical farming
              knowledge to help you succeed.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {[
              {
                icon: <Camera className="w-7 h-7 text-white" />,
                title: "Disease Detection",
                desc: "Upload plant images to instantly identify diseases and get treatment recommendations.",
                gradient: "from-green-500 to-emerald-600",
                bgGradient: "from-green-50 to-emerald-50",
              },
              {
                icon: <BarChart3 className="w-7 h-7 text-white" />,
                title: "Crop Analysis",
                desc: "Get detailed insights about your crops' health, growth patterns, and yield predictions.",
                gradient: "from-amber-500 to-orange-600",
                bgGradient: "from-amber-50 to-orange-50",
              },
              {
                icon: <Lightbulb className="w-7 h-7 text-white" />,
                title: "Smart Recommendations",
                desc: "Receive personalized advice on planting, fertilizing, and harvesting based on your data.",
                gradient: "from-blue-500 to-cyan-600",
                bgGradient: "from-blue-50 to-cyan-50",
              },
              {
                icon: <Shield className="w-7 h-7 text-white" />,
                title: "Weather Intelligence",
                desc: "Stay ahead with hyperlocal weather forecasts and climate-based farming recommendations.",
                gradient: "from-purple-500 to-pink-600",
                bgGradient: "from-purple-50 to-pink-50",
              },
            ].map((feature, i) => (
              <div
                key={i}
                className="group relative bg-white shadow-lg rounded-2xl p-6 hover:shadow-2xl transition-all duration-300 text-center border border-gray-100 hover:border-transparent hover:scale-105"
              >
                <div className={`absolute inset-0 bg-gradient-to-br ${feature.bgGradient} rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300`}></div>
                <div className="relative">
                  <div className={`w-16 h-16 bg-gradient-to-br ${feature.gradient} rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg group-hover:scale-110 transition-transform duration-300`}>
                    {feature.icon}
                  </div>
                  <h3 className="text-xl font-bold text-gray-900 mb-3">
                    {feature.title}
                  </h3>
                  <p className="text-gray-600 leading-relaxed">{feature.desc}</p>
                  <div className="mt-4">
                    <CheckCircle2 className="w-5 h-5 text-green-600 mx-auto" />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative bg-gradient-to-br from-green-600 via-emerald-600 to-blue-600 py-20 overflow-hidden">
        <div className="absolute top-0 left-0 w-96 h-96 bg-white/5 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-0 right-0 w-96 h-96 bg-white/5 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>

        <div className="relative max-w-7xl mx-auto px-6 lg:px-8 text-center text-white">
          <div className="inline-flex items-center gap-2 px-5 py-2 mb-6 bg-white/20 backdrop-blur-sm rounded-full border-2 border-white/30">
            <Zap className="w-5 h-5 text-white" />
            <span className="text-sm font-bold text-white">Start Your Journey Today</span>
          </div>

          <h2 className="text-4xl md:text-5xl font-bold mb-6">
            Ready to revolutionize your farming?
          </h2>
          <p className="text-xl text-white/90 mb-10 max-w-2xl mx-auto">
            Join thousands of farmers already using AgriFinSight to increase
            their yields with AI-powered insights.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              to="/register"
              className="group bg-white text-green-700 hover:bg-gray-50 px-10 py-4 text-lg font-bold rounded-xl inline-flex items-center justify-center transition-all shadow-2xl hover:shadow-3xl hover:scale-105"
            >
              Get Started
              <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link
              to="/ai-features"
              className="bg-white/10 backdrop-blur-sm border-2 border-white/30 text-white hover:bg-white/20 px-10 py-4 text-lg font-bold rounded-xl inline-flex items-center justify-center transition-all hover:scale-105"
            >
              Learn More
            </Link>
          </div>

          {/* Trust Indicators */}
          <div className="mt-12 flex flex-wrap items-center justify-center gap-8 text-white/80">
            <div className="flex items-center gap-2">
              <Star className="w-5 h-5 fill-yellow-400 text-yellow-400" />
              <Star className="w-5 h-5 fill-yellow-400 text-yellow-400" />
              <Star className="w-5 h-5 fill-yellow-400 text-yellow-400" />
              <Star className="w-5 h-5 fill-yellow-400 text-yellow-400" />
              <Star className="w-5 h-5 fill-yellow-400 text-yellow-400" />
              <span className="ml-2 font-semibold">4.9/5 from 10K+ reviews</span>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

