import React from "react";
import InsightWidget from "@/components/widgets/InsightWidget";
import { Navbar } from "@/components/Navbar";
import CO2SequestrationChart from "@/components/CO2SequestrationChart";

import mapImage from "../assets/map-img.png";
import vegImage from "../assets/veg-img.png";

const YearlyComparison = ({
    year,
    claimed,
    actual,
}: {
    year: string;
    claimed: string;
    actual: string;
}) => {
    return (
        <div className="flex flex-col">
            <div className="text-neutral-500 font-medium">{year}</div>
            <div className="flex justify-between w-full">
                <div>
                    <div className="text-lg font-medium text-yellow-600">
                        {claimed}
                    </div>
                    <div>Claimed</div>
                </div>
                <div>
                    <div className="text-lg font-medium text-emerald-600">
                        {actual}
                    </div>
                    <div className="text-right">Actual</div>
                </div>
            </div>
        </div>
    );
};

const mockData = {
    projectSummary:
        "The Kariba REDD+ Project, located in northwestern Zimbabwe, aims to reduce deforestation and enhance local livelihoods through sustainable land-use practices. Covering 784,987 hectares, the project involves community-based initiatives such as improved agriculture, beekeeping, and firewood plantations. The project is expected to generate 196,513,929 emission reductions over its lifetime.",
    riskMeter: {
        value: 84,
        label: "Unsafe",
    },
    projectLocation: {
        latitude: 27.234925,
        longitude: -18.936622,
    },
    projectGoals: {
        description: [
            "The project aims to reduce deforestation and forest degradation by implementing sustainable land-use practices while improving the livelihoods of local communities.",
            "It employs activities such as sustainable agriculture, fuelwood plantations, and alternative income sources to reduce reliance on forest resources. Additionally, funds generated from carbon credits will support local health and education initiatives.",
        ],
    },
    forestArea: {
        yearlyData: [
            {
                year: "2019",
                claimed: "3400 ha",
                claimedValue: 3000.0,
                actual: "2497.98 ha",
                actualValue: 2497.98,
            },
            {
                year: "2020",
                claimed: "3700 ha",
                claimedValue: 3700.0,
                actual: "2466.11 ha",
                actualValue: 2466.11,
            },
            {
                year: "2021",
                claimed: "4000 ha",
                claimedValue: 4000.0,
                actual: "2500 ha",
                actualValue: 2500.0,
            },
        ],
    },
    co2Reductions: {
        data: [
            {
                year: 2019,
                claimed: 8.9,
                actual: -6.302,
            },
            {
                year: 2020,
                claimed: 9.3,
                actual: 1.4697,
            },
            {
                year: 2021,
                claimed: 12.7,
                actual: -6.754,
            },
        ],
    },
    carbonContent: {
        calculated: "224.15 tC/ha",
        calculatedValue: 224.15,
        description: [
            "Carbon content is calculated based on the amount of carbon stored in the soil and vegetation of the area.",
            "This value is used to estimate the total carbon sequestration potential of the area.",
        ],
    },
    aiInsights: [
        "The project area has a high potential for carbon sequestration due to the presence of significant forest cover and biomass.",
        "There is a notable shortfall in achieving reforestation targets. This may be due to challenges such as deforestation pressures, poor enforcement of conservation measures, or external environmental factors.",
        "This indicates potential failures in conservation strategies or worsening deforestation trends despite the project’s intervention. External pressures such as illegal logging or land-use changes may have counteracted the expected carbon gains.",
        "The report states that “there is no significant income to offset deforestation mitigation costs without carbon revenues”—meaning project survival depends entirely on credit sales.",
    ],
};

const CarbonInsights: React.FC = () => {
    const {
        projectSummary,
        riskMeter,
        projectGoals,
        forestArea,
        co2Reductions,
        carbonContent,
        aiInsights,
    } = mockData;

    return (
        <div className="flex flex-col min-h-screen">
            <Navbar />
            <div className="h-[calc(100vh-50px)] p-8 overflow-hidden bg-gradient-to-b from-neutral-200 to-neutral-50 scrollbar-hide">
                <div className="mb-6">
                    <h1 className="scroll-m-20 text-3xl font-extrabold tracking-tighter lg:text-4xl mb-2 text-emerald-950">
                        Carbon Integrity Insights
                    </h1>
                    <h2 className="text-xl font-medium mb-4 text-neutral-500">
                        An AI-Driven Validation & Fraud Analysis Dashboard
                    </h2>
                </div>
                <div className="flex justify-between items-center gap-20 mb-8 w-full">
                    <div className="max-w-9/12">{projectSummary}</div>
                    <div className="mr-6">
                        <div className="w-full text-neutral-400 font-medium">
                            Fraud Risk Meter
                        </div>
                        <div className="flex items-baseline gap-2">
                            <div
                                className={`text-3xl font-semibold ${
                                    riskMeter.value < 25
                                        ? "text-green-600"
                                        : riskMeter.value < 50
                                        ? "text-yellow-600"
                                        : riskMeter.value < 75
                                        ? "text-orange-600"
                                        : "text-red-600"
                                }`}
                            >
                                {riskMeter.value}
                            </div>
                            <div className="text-sm mb-0.5">/ 100</div>
                        </div>
                        <div
                            className={`font-medium ${
                                riskMeter.value < 25
                                    ? "text-green-600"
                                    : riskMeter.value < 50
                                    ? "text-yellow-600"
                                    : riskMeter.value < 75
                                    ? "text-orange-600"
                                    : "text-red-600"
                            }`}
                        >
                            {riskMeter.value < 25
                                ? "Safe"
                                : riskMeter.value < 50
                                ? "Moderate"
                                : riskMeter.value < 75
                                ? "High"
                                : "Unsafe"}
                        </div>
                    </div>
                </div>

                <div className="grid grid-cols-12 grid-rows-2 gap-4">
                    {/* div1: spans rows 1-2, cols 1-3 */}
                    <div className="col-start-1 col-end-4 row-start-1 row-end-3">
                        <InsightWidget title="Project Goals">
                            <div className="flex flex-col gap-4 justify-evenly">
                                {projectGoals.description.map(
                                    (paragraph, index) => (
                                        <div
                                            key={index}
                                            className="text-sm"
                                        >
                                            {paragraph}
                                        </div>
                                    )
                                )}
                                <div className="rounded-md mt-2">
                                    <img
                                        className="rounded-lg"
                                        src={mapImage}
                                        alt="Project location map"
                                    />
                                </div>
                            </div>
                        </InsightWidget>
                    </div>
                    {/* div2: spans row 1, cols 4-6 */}
                    <div className="col-start-4 col-end-7 row-start-1 row-end-2">
                        <InsightWidget title="Vegetation Index">
                            <div className="rounded-md mt-2">
                                <img
                                    className="rounded-lg"
                                    src={vegImage}
                                    alt="Vegetation index visualization"
                                />
                            </div>
                        </InsightWidget>
                    </div>

                    {/* div3: spans row 2, cols 4-6 */}
                    <div className="col-start-4 col-end-7 row-start-2 row-end-3">
                        <InsightWidget title="Forest Area">
                            <div className="flex flex-col gap-2">
                                {forestArea.yearlyData.map((item, index) => (
                                    <YearlyComparison
                                        key={index}
                                        year={item.year}
                                        claimed={item.claimed}
                                        actual={item.actual}
                                    />
                                ))}
                            </div>
                        </InsightWidget>
                    </div>

                    {/* div4: spans row 1, cols 7-9 */}
                    <div className="col-start-7 col-end-10 row-start-1 row-end-2">
                        <InsightWidget title="CO₂ Reductions">
                            <CO2SequestrationChart data={co2Reductions.data} />
                        </InsightWidget>
                    </div>

                    {/* div5: spans row 1, cols 10-12 */}
                    <div className="col-start-10 col-end-13 row-start-1 row-end-2">
                        <InsightWidget title="Carbon Content">
                            <div className="flex flex-col gap-1">
                                <div className="text-neutral-500 font-medium">
                                    Calculated
                                </div>
                                <div className="text-emerald-600 text-xl font-medium">
                                    {carbonContent.calculated}
                                </div>
                                <div className="mt-4 flex flex-col gap-2">
                                    {carbonContent.description.map(
                                        (paragraph, index) => (
                                            <div key={index}>{paragraph}</div>
                                        )
                                    )}
                                </div>
                            </div>
                        </InsightWidget>
                    </div>

                    {/* div6: spans row 2, cols 7-12 */}
                    <div className="col-start-7 col-end-13 row-start-2 row-end-3">
                        <InsightWidget title="AI Insights">
                            {aiInsights.map((insight, index) => (
                                <div
                                    key={index}
                                    className="mb-2"
                                >
                                    💡 {insight}
                                </div>
                            ))}
                        </InsightWidget>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default CarbonInsights;
