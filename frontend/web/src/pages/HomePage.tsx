import { useEffect } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, Camera, BarChart3, Lightbulb, Shield } from "lucide-react";

// amCharts imports
import * as am5 from "@amcharts/amcharts5";
import * as am5map from "@amcharts/amcharts5/map";
import am5geodata_worldLow from "@amcharts/amcharts5-geodata/worldLow";
import am5themes_Animated from "@amcharts/amcharts5/themes/Animated";

export function HomePage() {
  useEffect(() => {
    // Root
    const root = am5.Root.new("chartdiv");
    root.setThemes([am5themes_Animated.new(root)]);

    // Chart
    const chart = root.container.children.push(
      am5map.MapChart.new(root, {
        panX: "rotateX",
        panY: "rotateY",
        projection: am5map.geoOrthographic(),
        paddingBottom: 20,
        paddingTop: 20,
        paddingLeft: 20,
        paddingRight: 20,
      })
    );

    // Polygon series
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

    // Highlight function
    const highlightCountry = (id: string) => {
      polygonSeries.mapPolygons.each((polygon) => {
        const countryId = polygon.dataItem?.get("id");
        polygon.setAll({
          fill:
            countryId === id ? am5.color(0xff5733) : am5.color(0xcccccc),
        });
      });
    };

    // Rotate to a country
    const selectCountry = (id: string) => {
      const dataItem = polygonSeries.getDataItemById(id);
      if (!dataItem) return;

      const target = dataItem.get("mapPolygon");
      if (target) {
        const centroid = target.geoCentroid();
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

    // Agric facts
    interface AgricFact {
      countryId: string;
      fact: string;
    }

    const agricFacts: AgricFact[] = [
      { countryId: "GH", fact: "🇬🇭 Ghana produces 25% of global cocoa." },
      { countryId: "NG", fact: "🇳🇬 Nigeria is the world’s largest yam producer." },
      { countryId: "BR", fact: "🇧🇷 Brazil is the largest coffee exporter." },
      { countryId: "ET", fact: "🇪🇹 Ethiopia is Africa’s biggest coffee producer." },
    ];

    const factBox = document.getElementById("factbox");
    let index = 0;

    const showNextFact = () => {
      const item = agricFacts[index];
      selectCountry(item.countryId);
      if (factBox) factBox.innerText = item.fact;
      index = (index + 1) % agricFacts.length;
    };

    showNextFact();
    const interval = setInterval(showNextFact, 5000);

    chart.appear(1000, 100);

    // Cleanup
    return () => {
      root.dispose();
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 to-secondary-50">
      {/* Hero Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-16">
        {/* 🌍 Globe Section */}
        <div className="flex flex-col items-center mb-10">
          <div id="chartdiv" className="w-full max-w-2xl h-[400px]" />
          <div
            id="factbox"
            className="mt-4 text-center text-xl font-semibold text-gray-800"
          ></div>
        </div>

        {/* Hero Text */}
        <div className="text-center">
          <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
            Smart Agriculture with
            <span className="text-primary-600"> AI-Powered</span> Insights
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            Transform your farming with intelligent disease detection, crop
            analysis, and personalized recommendations powered by machine
            learning.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              to="/register"
              className="btn btn-primary px-8 py-3 text-lg rounded-lg inline-flex items-center"
            >
              Get Started
              <ArrowRight className="ml-2 w-5 h-5" />
            </Link>
            <Link
              to="/login"
              className="btn btn-outline px-8 py-3 text-lg rounded-lg"
            >
              Sign In
            </Link>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            Everything you need for modern farming
          </h2>
          <p className="text-lg text-gray-600">
            Our platform combines cutting-edge AI with practical farming
            knowledge
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
          {[
            {
              icon: <Camera className="w-6 h-6 text-primary-600" />,
              title: "Disease Detection",
              desc: "Upload plant images to instantly identify diseases and get treatment recommendations",
              color: "primary",
            },
            {
              icon: <BarChart3 className="w-6 h-6 text-secondary-600" />,
              title: "Crop Analysis",
              desc: "Get detailed insights about your crops' health, growth patterns, and yield predictions",
              color: "secondary",
            },
            {
              icon: <Lightbulb className="w-6 h-6 text-green-600" />,
              title: "Smart Recommendations",
              desc: "Receive personalized advice on planting, fertilizing, and harvesting based on your data",
              color: "green",
            },
            {
              icon: <Shield className="w-6 h-6 text-purple-600" />,
              title: "Weather Integration",
              desc: "Stay ahead with weather forecasts and climate-based farming recommendations",
              color: "purple",
            },
          ].map((feature, i) => (
            <div key={i} className="card p-6 text-center">
              <div
                className={`w-12 h-12 bg-${feature.color}-100 rounded-lg flex items-center justify-center mx-auto mb-4`}
              >
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

      {/* CTA Section */}
      <div className="bg-primary-600 py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold text-white mb-4">
            Ready to revolutionize your farming?
          </h2>
          <p className="text-xl text-primary-100 mb-8">
            Join thousands of farmers already using AgriFinSight to increase
            their yields
          </p>
          <Link
            to="/register"
            className="btn bg-white text-primary-600 hover:bg-gray-100 px-8 py-3 text-lg rounded-lg inline-flex items-center"
          >
            Start Your Free Trial
            <ArrowRight className="ml-2 w-5 h-5" />
          </Link>
        </div>
      </div>
    </div>
  );
}

