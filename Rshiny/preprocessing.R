#### Preprocessing scripts
library("zoo")
library("reshape2")
source("c3aidatalake.R")
df = read_csv("countries_names.csv", col_types = list(col_character(),
                                                      col_character(),
                                                      col_character(),
                                                      col_character(),
                                                      col_double()))
states_abbreviations = read_csv("state_abbreviations.csv",
                                col_types = list(col_character(),
                                                 col_character()))

test_performed = read_csv("daily-tests-per-thousand-people-smoothed-7-day.csv")
test_performed$Date = as.Date(test_performed$Date, format = "%m/%d/%y")
mymonths <- c("jan","feb","mar",
              "april","may","june",
              "july","aug","sept",
              "oct","nov","dec")



fetch_all_prevalence <- function(country, region,  date){
  date0 <- date
  if (date > as.Date("2020-11-10")){
    date0 = as.Date("2020-11-10")
  }
  database_lookup = df[which((df$country_name == country) & (df$region == region)),]
  pop = database_lookup$pop
  database = database_lookup$database
  name = database_lookup$country
  abbr = ifelse (country == "United States", (states_abbreviations %>% filter(state == region))$state_abb, 
                 ifelse(((region == "American Samoa") ||(country == "Puerto Rico,U.S.")), country, ""))

  #### Fetch the FB data
  month = paste0(format(date0,"%m"), '_', mymonths[as.numeric(format(date0,"%m"))])
  fb_data = read_csv(paste0("FB_data/all_countries_", month,".csv"),
                     col_types = list(col_date(format = ""),
                                      col_character(), col_double(),
                                      col_double(),col_double(),col_character()))
  fb_data$region[which(is.na(fb_data$region))] = ""
  fb_data$region[which(fb_data$country =="Puerto Rico, U.S.")] = "Puerto Rico"
  fb_data$region[which(fb_data$country =="American Samoa")] = "American Samoa"
  fb_data$country[which(fb_data$country == "UnitedStates")] = "United States"
  fb_data$country[which(fb_data$country =="Puerto Rico, U.S.")] = "United States"
  fb_data$country[which(fb_data$country =="American Samoa")] = "United States"
  fb_data$region[which(fb_data$country =="Hong Kong")] = "Hong Kong"
  fb_data$country[which(fb_data$country == "Hong Kong")] = "China"
  fb_data$country[which(fb_data$country == "Antigua")] = "Antigua and Barbuda"
  fb_data$country[which(fb_data$country == "Palestine")] = "West Bank and Gaza"
  
  #### Find closest date
  #### Test rate
  if (country %in% unique(test_performed$Entity)){
    coldate = test_performed[which(test_performed$Entity == country),]
    x0 = which(abs(coldate$Date-date0) == min(abs(coldate$Date - date0)))
    if (length(x0)>1){
      x0 = x0[which(!is.null(coldate[x0,"Code"]))]
    }
    pct_tested = (coldate$new_tests_per_thousand_7day_smoothed[x0] * 1000)/pop
    #### Capture the uncertainty
    sd_pct_tested = sd(unlist(test_performed[which((test_performed$Entity == country) & (test_performed$Date > date0 -30) 
                                                   & (test_performed$Date < date0+30)),
                                             "new_tests_per_thousand_7day_smoothed"] * 1000/pop))
      
    }else{
    #### Try to get the test rate across the world
      coldate = test_performed
      x0 = which(abs(coldate$Date-date0) == min(abs(coldate$Date - date0)))
      if (length(x0)>1){
        x0 = x0[which(!is.null(coldate[x0,"Code"]))]
      }
      pct_tested = (mean(coldate$new_tests_per_thousand_7day_smoothed[x0]) * 1000)/pop
      #### Capture the uncertainty
      sd_pct_tested = sd(unlist(test_performed[which((test_performed$Date > date0-30) 
                                                     & (test_performed$Date < date0+30)),
                                               "new_tests_per_thousand_7day_smoothed"] * 1000/pop))
      
      
    }
  
  if (country %in% unique(fb_data$country)){
    coldate = fb_data[which(fb_data$country == country & fb_data$region == abbr),]
  }else{
    #### Try to get the test rate
    coldate = fb_data
  }
  x0 = which(abs(coldate$date-date0) == min(abs(coldate$date - date0)))
  sd_ili = sd(unlist(fb_data["pct_ili"]))
  sd_cli = sd(unlist(fb_data["pct_cli"]))
  sd_tested_fb = sd(unlist(fb_data["pct_tested"]))
  fb_data = fb_data[x0,]
  pct_ili = mean(fb_data$pct_ili)
  pct_cli = mean(fb_data$pct_cli)
  pct_tested_fb = mean(fb_data$pct_tested)

  
  
  #### Fetch cases on that day
  casecounts <- evalmetrics(
    "outbreaklocation",
    list(
      spec = list(
        ids = list(name),
        expressions = list(paste0(database, "_ConfirmedCases")),
        start = date0 - 31,
        end = date0,
        interval = "DAY"
      )
    )
  )
  casecounts = casecounts %>% mutate(new_cases = data - lag(data, default = 1)) %>% filter(dates >= date0 - 21)
  casecounts = casecounts %>% mutate(smoothed_cases=rollapply(new_cases, 7, mean, align='right',fill=NA))  ### thats the new cases counts smoothed over
  ### Let's now compute the prevalence: active cases over the last 14 days
  prevalence = sum(casecounts %>% dplyr::filter(dates > date0 -14) %>% select(smoothed_cases))/ pop
  sd_prev = sqrt(14) * sd(unlist(casecounts %>% dplyr::filter(dates > date0 -14) %>% select(smoothed_cases))/ pop)
  ### Thats the global prevalence in the population
  ### Now let's fetch the prevalence among the tested
  return(list(prev= prevalence, pct_tested = pct_tested, pct_tested_fb = pct_tested_fb, pct_ili = pct_ili, pct_cli= pct_cli, sd_pct_tested = sd_pct_tested,
              sd_pct_ili = sd_ili,sd_pct_cli = sd_cli, sd_pct_tested_fb = sd_tested_fb, sd_prev = sd_prev ))
}
