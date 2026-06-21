elasticity = {
    "water": {
        "supply": 0.4,
        "demand": -0.25 #Dalhulsen et al. 10.2307/3146872
    },
    "electricity": {
        "supply": 1,
        "demand": -0.7 #Zhu et al. 10.1016/j.jclepro.2018.08.027
    }
}

current_demand = {
    "LA": {
        "electricity": 64896e9, #64k GWh/yr, https://www.energy.ca.gov/data-reports/energy-almanac/california-electricity-data/california-energy-consumption-dashboards-0
        "water": 1489564 * 1.233e6 #L/yr https://ourcountyla.lacounty.gov/wp-content/uploads/2020/03/OurCounty-Indicators-Analysis.pdf?
    }
}

def price_change(current_price, old_demand, new_demand, supply_elasticity, demand_elasticity):
    return current_price * (new_demand / old_demand) / (supply_elasticity - demand_elasticity)